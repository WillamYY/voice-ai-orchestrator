"""Bounded four-worker pipeline with per-response cancellation and playback history."""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING

from src.core.events import AudioPacket, Response, SpeechSegment, SpeechStarted, UserTurn
from src.core.interfaces import ASRProvider, LLMProvider, RecorderProvider, TTSProvider
from src.core.segmenter import TextSegmenter

if TYPE_CHECKING:
    from src.config.settings import Settings

logger = logging.getLogger(__name__)


def _drain(q):
    while True:
        try:
            q.get_nowait()
            q.task_done()
        except queue.Empty:
            return


class ConversationManager:
    def __init__(
        self, *, recorder: RecorderProvider | None = None,
        asr: ASRProvider | None = None, llm: LLMProvider | None = None,
        tts: TTSProvider | None = None, config: Settings | None = None,
    ) -> None:
        if config is None:
            from src.config.settings import get_settings
            config = get_settings()
        self.config = config
        # Constructors allocate no hardware/model resources. prepare() owns startup.
        if recorder is None:
            from src.services.recorder import RecorderService
            recorder = RecorderService(config=config)
        if asr is None:
            from src.services.asr_service import FunASRService
            asr = FunASRService(config=config)
        if llm is None:
            from src.services.llm_service import LLMService
            llm = LLMService(config=config)
        if tts is None:
            from src.services.tts_service import EdgeTTSService
            tts = EdgeTTSService(config=config)
        
        self.recorder: RecorderProvider = recorder
        self.asr: ASRProvider = asr
        self.llm: LLMProvider = llm
        self.tts: TTSProvider = tts

        # Inter-Worker Bounded Queues
        self.q_audio = queue.Queue(maxsize=config.audio_queue_frames)   # Audio frames between mic capture and ASR
        self.q_text_asr = queue.Queue(maxsize=config.text_queue_size)   # Recognized user sentences passed from ASR to LLM
        self.q_text_llm = queue.Queue(maxsize=config.text_queue_size)   # Segmented LLM clauses passed to TTS for playback

        # Lifecycle & Concurrency Synchronization
        self._stop = threading.Event()        # Set when session is shutting down
        self._input_done = threading.Event()  # Set when audio capture finishes
        self._tts_ready = threading.Event()   # Barrier ensuring TTS mixer starts before mic
        self._lock = threading.RLock()        # Re-entrant lock for shared state mutations

        # State Tracking
        self._latest_utterance = 0            # Monotonically increasing ID per user speech turn
        self._response_id = 0                 # Monotonically increasing ID per assistant turn
        self._generation = 0                 # Advanced only by an explicit stop command
        self._active_response = None          # Current active Response instance
        self._asr_utterance = None            # Utterance ID currently being processed by ASR
        self._discard_utterance = False       # Flag to drop remainder of an error-prone utterance
        self.threads = []                     # Handles to the 4 worker threads
        self._provider_owners = {}
        self._cleanup_attempted = set()
        self.history = []                     # Message dictionaries with role/content, in conversation order
        self.errors = []                      # Recent error messages
        self.fatal_error = None
        self._closed = False
        self.running = False

    def _record_error(self, stage, error, *, fatal=False):
        message = f'{stage}: {error}'
        with self._lock:
            self.errors.append(message)
            self.errors = self.errors[-20:]  # Keep an in-memory rolling buffer of errors
            if fatal and self.fatal_error is None:
                self.fatal_error = message
        logger.error(message)
        if fatal:
            self._stop.set()
            self.cancel_response()

    def _spawn(self, name, target):
        """Wrap thread targets to capture unhandled exceptions as fatal pipeline errors."""
        def guarded():
            try:
                target()
            except Exception as error:
                self._record_error(name, error, fatal=True)
        thread = threading.Thread(name=name, target=guarded, daemon=True)
        for provider, worker in (('asr', self._asr_worker), ('llm', self._llm_worker),
                                 ('tts', self._tts_worker)):
            if target == worker:
                self._provider_owners[provider] = thread
        self.threads.append(thread)
        try:
            thread.start()
        except Exception:
            self.threads.remove(thread)
            self._provider_owners = {key: owner for key, owner in self._provider_owners.items()
                                     if owner is not thread}
            raise

    def _cleanup_action(self, label, action):
        """Report a failed cleanup without preventing other cleanup attempts."""
        try:
            action()
        except Exception as error:
            self._record_error(label, error, fatal=True)

    def _close_provider(self, name):
        # Normally called by the owning worker; shutdown handles providers whose
        # worker never started. Never close a device concurrently with its owner.
        with self._lock:
            if name in self._cleanup_attempted:
                return
            self._cleanup_attempted.add(name)
        self._cleanup_action(f'{name.upper()} cleanup', getattr(self, name).close)

    def start(self):
        """Public entrypoint of this application."""
        if self._closed or self.running:
            raise RuntimeError('Create a new manager for each session')
        self.running = True
        try:
            logger.info('Loading ASR and %s LLM before opening microphone...', self.config.llm_backend)
            # Warm up inference engines in the main thread prior to capture
            self.asr.prepare()
            self.llm.prepare()

            # Spawn TTS and wait for self._tts_ready barrier so audio device is initialized
            self._spawn('tts', self._tts_worker)
            while not self._tts_ready.wait(0.05):
                if self._stop.is_set():
                    raise RuntimeError(self.fatal_error or 'TTS startup failed')

            # Spawn remaining workers and audio producer
            self._spawn('asr', self._asr_worker)
            self._spawn('llm', self._llm_worker)
            self._spawn('capture', self._audio_producer)

            logger.info('Ready. Speak, use a configured stop phrase to cancel, or press Ctrl+C to exit.')
            # Main supervision loop: terminates on Ctrl+C, fatal error, or clean EOF drain
            while not self._stop.wait(0.05):
                if self._input_done.is_set() and self._queues_idle():
                    break
        except KeyboardInterrupt:
            logger.info('Stopping session')
        except Exception as error:
            self._record_error('Session', error, fatal=True)
        finally:
            self.shutdown()
        if self.fatal_error:
            raise RuntimeError(self.fatal_error)

    def _queues_idle(self):
        """Check if all bounded queues have processed all queued tasks."""
        for q in (self.q_audio, self.q_text_asr, self.q_text_llm):
            with q.all_tasks_done:
                if q.unfinished_tasks:
                    return False
        return True

    def cancel_response(self, reason='explicit stop'):
        """
        Signal cancellation to the currently active response.

        Each assistant response has its own cancellation Event token. 
        Late-arriving chunks from a cancelled response are safely dropped.
        """
        with self._lock:
            if self._active_response is not None and not self._active_response.cancelled.is_set():
                self._active_response.cancelled.set()
                logger.info('Response %s cancelled: %s', self._active_response.response_id, reason)

    def _begin_utterance(self):
        """Called immediately when speech activity is detected."""
        with self._lock:
            self._latest_utterance += 1
            # If aggressive barge-in is enabled, interrupt assistant the moment VAD trips
            if self.config.barge_in and self.config.barge_in_on_speech_start:
                self.cancel_response('speech-start detection (unconfirmed by ASR)')
            logger.info('Speech started: utterance %s', self._latest_utterance)
            return self._latest_utterance

    def stop_responses(self):
        """Cancel active and pre-stop pending replies, keeping future input usable."""
        with self._lock:
            self._generation += 1
            self.cancel_response('explicit stop')
            _drain(self.q_text_asr)

    def _handle_control_input(self, text: str) -> bool:
        """
        Consume empty input or an utterance containing a configured stop phrase.

        ASR calls this before queueing so cancellation never waits for the LLM
        worker. Direct text callers use the same guard and command vocabulary.
        """
        if not text.strip():
            return True
        normalized = text.casefold()
        if any(phrase.strip().casefold() in normalized for phrase in self.config.stop_phrases):
            self.stop_responses()
            return True
        return False

    def _put(self, q, item, cancel_event=None):
        """Bounded queue put helper that aborts if the session stops or response is cancelled."""
        deadline = time.monotonic() + self.config.downstream_timeout
        while not self._stop.is_set():
            if cancel_event is not None and cancel_event.is_set():
                return False
            try:
                q.put(item, timeout=0.05)
                return True
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Downstream queue stalled beyond DOWNSTREAM_TIMEOUT')
        return False

    def _audio_producer(self):
        """Mic reader and VAD event generator."""
        utterance_id = None
        try:
            with self.recorder as mic:
                for chunk in mic.stream():
                    if self._stop.is_set():
                        break
                    # SpeechStarted token indicates transition from silence to voice
                    if isinstance(chunk, SpeechStarted):
                        utterance_id = self._begin_utterance()
                        continue
                    if utterance_id is None:
                        if chunk is None:
                            continue
                        utterance_id = self._begin_utterance()
                    try:
                        # Forward audio chunk tagged with utterance ID
                        self.q_audio.put_nowait(AudioPacket(utterance_id, chunk))
                    except queue.Full as error:
                        # Fast-fail rather than dropping audio and losing ASR synchronization
                        raise RuntimeError('Audio queue overflow; session stopped rather than dropping speech. Check ASR speed/device.') from error
                    # chunk is None indicates silence threshold reached (end of speech)
                    if chunk is None:
                        utterance_id = None
        finally:
            self._input_done.set()

    def _process_audio_chunk(self, packet):
        """Unpack AudioPacket and drive the streaming ASR service."""
        if not isinstance(packet, AudioPacket):
            packet = AudioPacket(self._latest_utterance, packet)
        
        # New utterance started: reset ASR state/cache
        if packet.utterance_id != self._asr_utterance:
            self.asr.reset_state()
            self._asr_utterance = packet.utterance_id
            self._discard_utterance = False
            
        try:
            if packet.data is None:
                # None marks utterance endpoint
                if self._discard_utterance:
                    self.asr.reset_state()
                else:
                    text = self.asr.finish_utterance().strip()
                    if text:
                        logger.info('User: %s', text)
                        # Immediate intercept for verbal stop commands
                        with self._lock:
                            if self._handle_control_input(text):
                                return
                            # Standard barge-in: cancel existing response when user completes new turn
                            if self.config.barge_in:
                                self.cancel_response('new completed user utterance')
                            turn = UserTurn(packet.utterance_id, text, self._generation)
                        # A concurrent stop may invalidate this turn before put completes;
                        # the consumer checks its generation even if it arrives after draining.
                        self._put(self.q_text_asr, turn)
            elif not self._discard_utterance:
                # Feed streaming audio bytes into ASR
                self.asr.transcribe(packet.data)
        except Exception:
            self._discard_utterance = True
            self.asr.reset_state()
            raise
        finally:
            if packet.data is None:
                self._discard_utterance = False

    def _asr_worker(self):
        """Consumes raw audio packets from q_audio."""
        try:
            while not self._stop.is_set():
                try:
                    packet = self.q_audio.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    self._process_audio_chunk(packet)
                except Exception as error:
                    self._record_error('ASR utterance discarded', error)
                finally:
                    self.q_audio.task_done()
        finally:
            self._close_provider('asr')

    def _process_user_turn(self, turn):
        """Query LLM and break streamed tokens into sentences."""
        with self._lock:
            if isinstance(turn, str):
                turn = UserTurn(self._latest_utterance, turn, self._generation)
            if turn.generation != self._generation or self._stop.is_set():
                return
            if self._handle_control_input(turn.text):
                return
            self._response_id += 1
            # Instantiate a fresh Response lifecycle token for this specific reply
            response = Response(self._response_id, generation=turn.generation)
            self._active_response = response
            
            # Clone history and inject system prompt for the prompt context
            prior = [dict(m) for m in self.history]
            if self.config.system_prompt.strip():
                prior.insert(0, {'role': 'system', 'content': self.config.system_prompt})
            self.history.append({'role': 'user', 'content': turn.text})

        segmenter = TextSegmenter(config=self.config)
        generator = None
        first_token = True
        logger.info('LLM request started: response %s, utterance %s', response.response_id, turn.utterance_id)
        
        try:
            if response.cancelled.is_set() or self._stop.is_set():
                return
            generator = self.llm.generate_response(turn.text, prior, cancel_event=response.cancelled)
            for token in generator:
                if response.cancelled.is_set() or self._stop.is_set():
                    break
                if token and first_token:
                    logger.info('LLM first text received: response %s', response.response_id)
                    first_token = False
                
                # The segmenter accumulates raw chunks and cleans completed segments.
                for segment in segmenter.feed(token or ''):
                    if not self._put(self.q_text_llm, SpeechSegment(response, segment), response.cancelled):
                        break

            # Push any trailing characters left in the segmenter buffer
            if not response.cancelled.is_set() and not self._stop.is_set():
                for segment in segmenter.finish():
                    self._put(self.q_text_llm, SpeechSegment(response, segment), response.cancelled)
                # Sentinel packet (text=None) signals TTS that no more segments are coming for this response
                self._put(self.q_text_llm, SpeechSegment(response, None), response.cancelled)
                
                # Block until TTS completes playback of the entire sentence queue
                deadline = time.monotonic() + self.config.downstream_timeout
                played_count = len(response.played)
                while not response.done.wait(0.05):
                    if response.cancelled.is_set() or self._stop.is_set():
                        break
                    with self._lock:
                        if len(response.played) != played_count:
                            played_count = len(response.played)
                            deadline = time.monotonic() + self.config.downstream_timeout
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Playback made no progress within DOWNSTREAM_TIMEOUT')
        except Exception:
            response.cancelled.set()
            raise
        finally:
            if generator is not None:
                close = getattr(generator, 'close', None)
                if close:
                    self._cleanup_action('LLM stream cleanup', close)

            # Accurate History Synchronization
            with self._lock:
                # Commit only fully completed playback segments (or diagnostic output).
                # An interrupted segment is omitted, even if some words were audible.
                spoken = ''.join(response.played).strip()
                if spoken:
                    self.history.append({'role': 'assistant', 'content': spoken})
                
                # Prune history to maintain bounded sliding context window
                user_indexes = [i for i, m in enumerate(self.history) if m['role'] == 'user']
                if len(user_indexes) > self.config.history_turns:
                    self.history = self.history[user_indexes[-self.config.history_turns]:]
                
                if self._active_response is response:
                    self._active_response = None
            logger.info('Response %s finished: cancelled=%s, played_segments=%s',
                        response.response_id, response.cancelled.is_set(), len(response.played))

    def _llm_worker(self):
        """Pull completed UserTurns and invokes generation."""
        try:
            while not self._stop.is_set():
                try:
                    turn = self.q_text_asr.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    self._process_user_turn(turn)
                except Exception as error:
                    self._record_error('LLM response cancelled', error)
                finally:
                    self.q_text_asr.task_done()
        finally:
            self._close_provider('llm')

    def _tts_worker(self):
        """Synthesize text segments and stream to audio playback."""
        try:
            self.tts.prepare()
            self._tts_ready.set()  # Signal to start() that audio device is ready
            while not self._stop.is_set():
                try:
                    segment = self.q_text_llm.get(timeout=0.05)
                except queue.Empty:
                    continue
                response = segment.response
                try:
                    with self._lock:
                        if response.generation != self._generation:
                            response.cancelled.set()
                    # None is the end-of-response delimiter
                    if segment.text is None:
                        response.done.set()
                    elif not response.cancelled.is_set():
                        # Synchronous playback of this segment; stop early if cancel_event is set
                        played = self.tts.speak(segment.text, cancel_event=response.cancelled)
                        with self._lock:
                            if played and not response.cancelled.is_set():
                                response.played.append(segment.text)
                                logger.info('Assistant: %s', segment.text.strip())
                            elif not response.cancelled.is_set():
                                raise RuntimeError('TTS did not complete playback')
                except Exception as error:
                    response.error = str(error)
                    response.cancelled.set()
                    response.done.set()
                    self._record_error('TTS response cancelled', error)
                finally:
                    self.q_text_llm.task_done()
        except Exception as error:
            # Record the worker failure before cleanup can report another error.
            self._record_error('TTS worker', error, fatal=True)
            raise
        finally:
            self._close_provider('tts')

    def shutdown(self):
        """Coordinate clean termination of all threads, audio streams, and queues."""
        if self._closed:
            return
        self._closed = True
        self.running = False
        
        # Trigger cooperative stop events
        self._stop.set()
        self.cancel_response()
        self._cleanup_action('Recorder stop', self.recorder.stop)
        self._cleanup_action('TTS stop', self.tts.stop)

        # Join all worker threads against configured shutdown timeout
        deadline = time.monotonic() + self.config.shutdown_timeout
        for thread in self.threads:
            if thread is not threading.current_thread():
                self._cleanup_action(f'Join {thread.name}',
                    lambda thread=thread: thread.join(timeout=max(0, deadline - time.monotonic())))

        # Report any deadlocked or runaway threads
        alive = [t.name for t in self.threads if t.is_alive()]
        for name in ('asr', 'llm', 'tts'):
            owner = self._provider_owners.get(name)
            if owner is None or not owner.is_alive():
                self._close_provider(name)
        if alive:
            self._record_error('Shutdown', f'Native/provider calls still running: {alive}; restart the process', fatal=True)
        else:
            # Drain residual queues to prevent thread join locks on task_done
            for q in (self.q_audio, self.q_text_asr, self.q_text_llm):
                _drain(q)
        if alive:
            logger.error('Session shutdown incomplete: workers still running')
        else:
            logger.info('Session closed with errors' if self.fatal_error else 'Session closed')
