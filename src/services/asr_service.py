"""Adapter for mono 16 kHz PCM16 and Paraformer streaming text deltas."""

import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


class FunASRService:
    def __init__(self, *, model=None, config=None):
        if config is None:
            from src.config.settings import get_settings
            config = get_settings()
            
        # Paraformer acoustic models are trained strictly on 16 kHz audio
        if config.sample_rate != 16000:
            raise ValueError("Paraformer streaming requires 16000 Hz input; resample before ASR")
            
        self._config = config
        self._model = model
        
        # Paraformer Streaming Stride Configuration
        # Format: [history_frames, current_chunk_frames, lookahead_frames]
        # Recall: samples = sample_rate (Hz) * duration (s)
        # 1 acoustic frame = 960 audio samples = 60 ms duration
        self._chunk_size = [0, 10, 5]
        
        # Minimum byte size required to trigger an intermediate decode step:
        # 10 frames * 960 samples/frame * 2 bytes/sample (16-bit PCM) = 19,200 bytes
        self._min_feed_size = self._chunk_size[1] * 960 * 2
        
        # Initialize internal utterance cache and audio buffers
        self.reset_state()

    def _load_model(self):
        """Lazy-load the FunASR AutoModel onto the selected inference device (CPU/CUDA)."""
        if self._model is None:
            # Route model weights to a custom cache directory
            os.environ.setdefault('MODELSCOPE_CACHE', self._config.asr_cache_dir)
            from funasr import AutoModel
            from src.services.device import inference_device
            
            self._model = AutoModel(
                model=self._config.asr_model,
                device=inference_device(self._config.use_gpu),
                disable_update=True,    # Disable FunASR package update checks; model downloads remain possible
                disable_pbar=True,  # Turn off terminal progress bars
                verbose=False,  # Suppress FunASR's internal debug prints
            )

    def prepare(self) -> None:
        """Load weights into memory during manager startup."""
        self._load_model()

    def _decode(self, data, *, is_final):
        """Convert raw PCM bytes into normalized float32 and invoke the neural network."""
        self._load_model()
        
        # Convert little-endian signed 16-bit integers (<i2) to float32 normalized to [-1.0, 1.0]
        # Dividing by 32768.0 converts standard PCM16 amplitudes to floating-point audio
        audio = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.0
        
        # Generate Δtext using the model's internal streaming cache
        results = self._model.generate(
            input=audio,
            cache=self._cache,
            is_final=is_final,
            chunk_size=self._chunk_size,
        )

        # Reconstruct the transcript text from current audio slices
        # Paraformer returns decoded Δtext, not the full cumulative hypothesis
        # Δtext from streaming mode: only newly recognized texts (including whitespace) by the current chunk
        text = "".join(result.get("text", "") for result in (results or []))
        self._transcript.append(text)
        return text

    def transcribe(self, audio_chunk: bytes) -> str:
        """
        Accept aligned PCM16 bytes and return newly decoded text.
        Decode only when more than one stride (_min_feed_size) is buffered,
        retaining real audio for the final decode at the utterance endpoint.
        """
        try:
            # PCM16 requires pairs of 2 bytes per sample; an odd byte count indicates data corruption
            if len(audio_chunk) % 2:
                raise ValueError("PCM16 chunks must contain complete two-byte samples")
                
            self._audio_buffer.extend(audio_chunk)
            parts = []
            
            # Drain full strides while retaining up to one complete stride for finalization
            while len(self._audio_buffer) > self._min_feed_size:
                data = bytes(self._audio_buffer[:self._min_feed_size])
                del self._audio_buffer[:self._min_feed_size]
                parts.append(self._decode(data, is_final=False))
                
            return "".join(parts)
        except Exception:
            # Reset cache and buffers on unhandled decoder exceptions to avoid poisoning next turn
            self.reset_state()
            raise

    def finish_utterance(self) -> str:
        """Flush residual audio with is_final=True and return the full utterance transcript."""
        try:
            # If any trailing audio remains in the buffer, decode it as the final chunk
            if self._audio_buffer:
                self._decode(bytes(self._audio_buffer), is_final=True)
            # Join all collected segments into one complete sentence
            return "".join(self._transcript).strip()
        finally:
            # Guarantee state cleanup so the next utterance starts with a pristine cache
            self.reset_state()

    def reset_state(self) -> None:
        """
        Discard active utterance audio and clear the acoustic cache without decoding.
        Used at utterance boundaries, on decode errors, and during teardown.
        Barge-in cancels the assistant response, not the user's ASR state.
        """
        self._cache = {}             # Clear Paraformer's recurrent acoustic context
        self._audio_buffer = bytearray()  # Clear un-decoded audio bytes
        self._transcript = []        # Clear delta transcript pieces

    def close(self) -> None:
        """Release state buffers during service teardown."""
        self.reset_state()
