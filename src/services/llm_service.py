"""Cancellable local/API generation with bounded output and error propagation."""
import asyncio
from collections.abc import Iterator
import logging
import queue
import threading
import time

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, *, config=None):
        if config is None:
            from src.config.settings import get_settings
            config = get_settings()
        self.config = config
        self.mode = config.llm_backend  # 'local' or 'api'
        self._model = self._tokenizer = None
        self._worker = None            # Background worker thread running generation
        self._cancel = None            # Active cancellation token
        self._closed = False

    def prepare(self) -> None:
        """Pre-load local Hugging Face weights and tokenizer onto the inference device."""
        if self.mode == 'local' and self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from src.services.device import inference_device
            
            device = inference_device(self.config.use_gpu)
            logger.info('Loading local LLM on %s', device)

            # Initialize the tokenizer
            # local_files_only=True ensures the pipeline runs offline without hitting Hugging Face Hub
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model_path, local_files_only=True
            )
            
            # Select fp32 for CPU for compatibility
            dtype = 'auto' if device == 'cuda' else torch.float32
            # Load the actual NN weights
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model_path, dtype=dtype,
                local_files_only=True
            ).to(device).eval()

    def _messages(self, user_input, history):
        """Construct an OpenAI-style messages list."""
        # Shallow copy the history to prevent it from being modified
        return [dict(m) for m in history] + [{'role': 'user', 'content': user_input}]

    def _local_inputs(self, messages):
        """Calculate input budget, format chat templates, and prune old context turns."""
        # Determine available prompt budget without overflowing model context length
        budget = min(
            self.config.llm_input_tokens,
            self._model.config.max_position_embeddings - self.config.llm_max_tokens
        )
        messages = list(messages)

        # Keep trying to fit the inputs into the token budget
        while True:
            # Format prompt using the tokenizer's chat template (e.g. ChatML for Qwen)
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors='pt', add_special_tokens=False)

            # If fit within the budget, return successfully
            if inputs['input_ids'].shape[-1] <= budget:
                return inputs.to(self._model.device)
                
            # If exceed budget, prune the oldest user-led turn pair
            # Retain system instructions (index 0 if present) and the current query (last index)
            first = next((i for i, m in enumerate(messages[:-1]) if m['role'] != 'system'), None)
            if first is None:
                raise ValueError('Current utterance/system prompt exceeds LLM input token budget')
                
            # Delete one full turn (user prompt + assistant response)
            end = first + 1
            while end < len(messages) - 1 and messages[end]['role'] != 'user':
                end += 1
            del messages[first:end]

    async def _api_generate(self, messages, emit_async, cancel):
        """Stream completions from OpenAI-compatible HTTP servers."""
        from openai import AsyncOpenAI
        
        async with AsyncOpenAI(
            base_url=self.config.llm_api_url,
            api_key=self.config.llm_api_key,
            timeout=self.config.request_timeout,
            max_retries=0  # Fail fast rather than hanging audio pipeline on retries
        ) as client:
            async def request():
                """Pure streaming logic."""
                stream = await client.chat.completions.create(
                    model=self.config.llm_api_model,
                    messages=messages,
                    stream=True,
                    max_tokens=self.config.llm_max_tokens
                )
                async with stream:
                    async for chunk in stream:
                        if chunk.choices:
                            text = chunk.choices[0].delta.content
                            if text:
                                await emit_async(text)

            # Create a separate async task for streaming loop
            task = asyncio.create_task(request())
            try:
                # Poll loop checking cooperative cancellation while waiting for stream
                while not task.done():
                    if cancel.is_set():
                        return
                    await asyncio.wait({task}, timeout=0.05)
                await task
            finally:
                # Cancel network stream task if cancelled or aborted early
                if not task.done():
                    task.cancel()
                # Await cancellation completion while suppressing CancelledError to keep logs clean
                await asyncio.gather(task, return_exceptions=True)

    def _local_generate(self, messages, emit, cancel):
        """Run Hugging Face local model inference with token-by-token streaming."""
        import torch
        from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
        
        inputs = self._local_inputs(messages)

        # Inject cancellation check into PyTorch's internal token generation loop
        class CancelCriteria(StoppingCriteria):
            # StoppingCriteria: an interface provided by Hugging Face to inject arbitrary break conditions
            # Hugging Face calls __call__ method after every single token iteration
            def __call__(self, input_ids, scores, **kwargs):
                return cancel.is_set()

        # Bridge model output tokens to the emit queue callback
        class OutputStreamer(TextStreamer):
            # TextStreamer calls on_finalized_text method only when accumulated token IDs coalesce into valid, printable strings
            def on_finalized_text(self, text, stream_end=False):
                if text:
                    emit(text)

        streamer = OutputStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            self._model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=self.config.llm_max_tokens,
                stopping_criteria=StoppingCriteriaList([CancelCriteria()])
            )

    def generate_response(
        self, user_input: str, history: list[dict[str, str]], *,    # Bare * is just a delimiter; everything after * must be keyword-only.
        cancel_event: threading.Event | None = None,
    ) -> Iterator[str]:
        """Main public generator yielding tokens from a background inference worker."""
        # Check preconditions
        if self._closed:
            raise RuntimeError('LLM service is closed')
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError('Previous inference is still stopping; restart if it does not recover')
        # Ensure a valid threading.Event exists even if caller did not supply one
        cancel = cancel_event if cancel_event is not None else threading.Event()
        if cancel.is_set():
            return

        # Initialize channel
        self.prepare()
        messages = self._messages(user_input, history)
        output = queue.Queue(maxsize=32)  # Bounded queue pauses producer if consumer lags too much
        done = threading.Event()          # Signal inference worker completed
        errors = []
        progress_lock = threading.Lock()
        last_progress = time.monotonic()
        blocked_since = None
        self._cancel = cancel

        def check_deadlines():
            if done.is_set() or cancel.is_set():
                return
            with progress_lock:
                now = time.monotonic()
                if blocked_since is not None:
                    if now - blocked_since >= self.config.downstream_timeout:
                        raise TimeoutError('LLM output queue stalled beyond DOWNSTREAM_TIMEOUT')
                elif now - last_progress >= self.config.llm_timeout:
                    raise TimeoutError('LLM produced no text within LLM_TIMEOUT')

        def enqueue(text):
            nonlocal last_progress, blocked_since
            check_deadlines()
            with progress_lock:
                try:
                    output.put_nowait(text)
                except queue.Full:
                    if blocked_since is None:
                        blocked_since = time.monotonic()
                    return False
                # Time waiting for consumer capacity is not inference idle time.
                last_progress = time.monotonic()
                blocked_since = None
                return True

        def emit(text):
            """Thread-safe queue producer for synchronous generation mode."""
            while not cancel.is_set():
                if enqueue(text):
                    return
                cancel.wait(0.02)

        async def emit_async(text):
            """Thread-safe queue producer for asynchronous generation mode."""
            while not cancel.is_set():
                if enqueue(text):
                    return
                await asyncio.sleep(0.02)

        def produce():
            """Worker target executing either API async event loop or local PyTorch inference."""
            try:
                if self.mode == 'api':
                    asyncio.run(self._api_generate(messages, emit_async, cancel))
                else:
                    self._local_generate(messages, emit, cancel)
                check_deadlines()
            except Exception as error:
                errors.append(error)
            finally:
                done.set()

        # Spawn background daemon worker so generator caller can monitor timeouts and cancellations
        self._worker = threading.Thread(target=produce, name='llm-inference', daemon=True)
        self._worker.start()
        
        try:
            while not cancel.is_set():
                if errors:
                    if isinstance(errors[0], TimeoutError):
                        raise errors[0]
                    raise RuntimeError(f'LLM provider failed: {errors[0]}') from errors[0]
                check_deadlines()
                try:
                    text = output.get(timeout=0.05)
                except queue.Empty:
                    if done.is_set():
                        if errors:
                            if isinstance(errors[0], TimeoutError):
                                raise errors[0]
                            raise RuntimeError(f'LLM provider failed: {errors[0]}') from errors[0]
                        break
                    continue
                if not cancel.is_set():
                    yield text
        finally:
            # Stop the worker if caller abandons generator early without cancelling
            if not done.is_set():
                cancel.set()
            self._worker.join(timeout=self.config.shutdown_timeout)
            if self._worker.is_alive():
                logger.error('Inference did not stop within shutdown timeout')

    def close(self) -> None:
        """Shuts down active inference workers and closes service."""
        self._closed = True
        if self._worker is not None and self._worker.is_alive():
            if self._cancel is not None:
                self._cancel.set()
            self._worker.join(timeout=self.config.shutdown_timeout)
            if self._worker.is_alive():
                raise RuntimeError('Native LLM inference is still running; restart the process')
