"""Microphone capture and VAD events, independent of ASR/model inference."""
import collections
import math
import logging
import threading
from collections.abc import Iterator
from typing import Self
from src.core.events import SpeechStarted

logger = logging.getLogger(__name__)


class RecorderService:
    def __init__(self, *, config=None):
        if config is None:
            from src.config.settings import get_settings
            config = get_settings()
        self.config = config
        
        # Audio format configuration (16 kHz, mono PCM16)
        self.rate = config.sample_rate
        # Equation: Duration = samples / sample_rate
        # Frame size in samples: e.g., 0.02 * 16000 = 320 samples (640 bytes)
        self.chunk_size = round(self.rate * config.chunk_duration)
        
        # Cooperative cancellation token for stopping the recording loop
        self._stop = threading.Event()
        self._pa = None      # PyAudio instance
        self._stream = None  # PyAudio input stream
        self._vad = None     # webrtcvad.Vad instance

    def __enter__(self) -> Self:
        """Context manager entry: open the mic hardware device and prepare VAD."""
        import pyaudio
        import webrtcvad
        self._stop.clear()
        self._pa = pyaudio.PyAudio()
        try:
            # VAD aggressiveness: 0 (least restrictive) to 3 (most restrictive speech detection).
            # Classification only: this does not remove noise from the audio.
            self._vad = webrtcvad.Vad(self.config.vad_mode)
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.config.input_device_index,
            )
        except Exception as startup_error:
            # Ensure resources are cleaned up if device acquisition fails
            try:
                self.__exit__(None, None, None)
            except Exception as cleanup_error:
                raise RuntimeError(f'Microphone startup failed: {startup_error}; cleanup: {cleanup_error}') from startup_error
            raise
        return self

    def __exit__(self, *_: object) -> None:
        """Context manager exit: guarantee audio stream closure and device teardown."""
        self._stop.set()
        stream, pa = self._stream, self._pa
        self._stream = self._pa = None
        errors = []
        actions = []
        if stream is not None:
            actions.extend([('stop stream', stream.stop_stream), ('close stream', stream.close)])
        if pa is not None:
            actions.append(('terminate PyAudio', pa.terminate))
        for label, action in actions:
            try:
                action()
            except Exception as error:
                errors.append(f'{label}: {error}')
        if errors:
            raise RuntimeError('Microphone cleanup failed: ' + '; '.join(errors))

    def stop(self) -> None:
        """Signal the capture generator to terminate on the next frame boundary."""
        self._stop.set()

    def stream(self) -> Iterator[SpeechStarted | bytes | None]:
        """Main public generator yielding speech start tokens, audio bytes, and endpoint markers."""
        if self._stream is None:
            raise RuntimeError('Microphone is not open')
        yield from self._detect(self._frames())

    def _frames(self):
        """Pull audio chunks from the hardware buffer."""
        while not self._stop.is_set():
            # exception_on_overflow=True ensures that if downstream processing stalls,
            # we surface an error rather than silently stitching dropped audio
            yield self._stream.read(self.chunk_size, exception_on_overflow=True)

    def _detect(self, frames):
        """Pure VAD state machine and domain logic to classify incoming frames into utterance events."""
        duration = self.config.chunk_duration
        
        # Frame count thresholds based on timing configurations:
        # Minimum consecutive voiced frames required to confirm speech onset (e.g., 100ms / 20ms = 5 frames)
        start_frames = math.ceil(self.config.speech_start_ms / (duration * 1000))
        # Silence frames required to conclude an utterance (e.g., 800ms / 20ms = 40 frames)
        silence_limit = math.ceil(self.config.silence_threshold / duration)
        # Safety cutoff to prevent runaway recordings (e.g., 30s / 0.02s = 1500 frames)
        max_frames = math.ceil(self.config.max_utterance_seconds / duration)
        # Hold ~200ms of pre-voice frames prior to VAD threshold confirmation
        # In addition to the confirmation window; preceding audio is not guaranteed
        # silence. At startup, only the history already captured is available.
        pre_vad_frames = math.ceil(0.2 / duration)

        pre_record_buffer = collections.deque(maxlen=start_frames + pre_vad_frames)
        
        active = False                  # Current speech state: True = user is speaking
        voiced_run = 0                  # Consecutive voiced frames accumulator
        silent_run = 0                  # Consecutive silent frames accumulator
        turn_frames = 0                 # Total frames in the current active turn

        for chunk in frames:
            if self._stop.is_set():
                return

            # Check if this PCM slice contains human voice
            voiced = self._vad.is_speech(chunk, self.rate)

            if not active:
                # State: Inactive (Listening)
                pre_record_buffer.append(chunk)
                voiced_run = voiced_run + 1 if voiced else 0

                # When voiced frames cross the threshold, transition to Active state
                if voiced_run >= start_frames:
                    active = True
                    turn_frames = len(pre_record_buffer)
                    silent_run = 0
                    
                    # Notify listeners that speech has begun (could trigger barge-in in manager if enabled)
                    yield SpeechStarted()
                    # Flush pre-roll buffer to avoid clipping the start of words
                    yield from pre_record_buffer
                    pre_record_buffer.clear()
            else:
                # State: Active (Recording)
                yield chunk
                turn_frames += 1
                silent_run = 0 if voiced else silent_run + 1

                # Check if speech ended via silence timeout or maximum duration limit
                if silent_run >= silence_limit or turn_frames >= max_frames:
                    reason = 'silence' if silent_run >= silence_limit else 'maximum duration'
                    logger.info('Speech endpoint: %s, duration %.2fs', reason, turn_frames * duration)
                    if reason == 'maximum duration':
                        logger.warning(
                            'No silence endpoint within %.1fs; check microphone/noise and SILENCE_THRESHOLD',
                            self.config.max_utterance_seconds
                        )
                    
                    # Signal the end of this utterance to ASR
                    yield None
                    
                    # Reset state machine for the next user utterance
                    active = False
                    voiced_run = silent_run = turn_frames = 0

        # If loop exited while an utterance was still active, emit an endpoint
        if active and not self._stop.is_set():
            yield None
