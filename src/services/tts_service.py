"""Cancellable sentence synthesis and playback, all audio I/O owned by TTS worker."""
import asyncio
from io import BytesIO
import threading
import time

class EdgeTTSService:
    def __init__(self, *, config=None, mixer=None, communicate_factory=None):
        if config is None:
            from src.config.settings import get_settings
            config = get_settings()
        self.config = config

        # Audio driver and network client injection handles
        self._mixer = mixer
        self._communicate = communicate_factory

        self._current = None    # Active threading.Event for the currently synthesizing sentence
        self._lock = threading.Lock()   # Synchronize state mutations across threads (speak vs stop vs close)
        
        # Service lifecycle flags
        self._closed = False
        self._prepared = False
        self._mixer_needs_close = False

    def prepare(self) -> None:
        """
        Pre-load heavy libraries and initialize sound hardware.
        """
        if self._prepared:
            return
        if self._mixer is None:
            import pygame
            self._mixer = pygame.mixer
        if self._communicate is None:
            import edge_tts
            self._communicate = edge_tts.Communicate

        # Initialize the underlying SDL audio driver used by pygame
        self._mixer_needs_close = True  # init() can acquire resources before failing.
        self._mixer.init()
        self._prepared = True

    async def _collect(self, text) -> bytes:
        """Stream MP3 chunks over WebSocket asynchronously from Microsoft Edge TTS."""
        audio = bytearray()
        
        # Connect to EdgeTTS WebSocket stream endpoint
        async for part in self._communicate(text, self.config.tts_model).stream():
            if part['type'] == 'audio':
                # Out-of-memory (OOM) protection
                if len(audio) + len(part['data']) > self.config.tts_max_segment_bytes:
                    raise RuntimeError(
                        f'TTS segment exceeded {self.config.tts_max_segment_bytes} bytes limit'
                    )
                audio.extend(part['data'])

        # Check if receive audio from TTS successfully            
        # edge_tts might return only metadata frames if the voice model rejected the text
        if not audio:
            raise RuntimeError('TTS returned no audio')
            
        return bytes(audio)

    async def _synthesize(self, text, cancel) -> bytes | None:
        """
        Supervise network synthesis with timeout protection, cooperative cancellation, and retries.
        """
        for attempt in range(self.config.tts_max_retries + 1):
            # Spawn network streaming into a decoupled concurrent asyncio task
            task = asyncio.create_task(self._collect(text))
            deadline = time.monotonic() + self.config.tts_timeout
            
            try:
                # Poll loop checking cooperative cancellation while waiting for stream
                while not task.done():
                    # Abort immediately if user started speaking
                    if cancel.is_set():
                        return None
                        
                    # Guard against hung TCP sockets or cold-start edge_tts delays
                    if time.monotonic() >= deadline:
                        raise TimeoutError('TTS synthesis timeout')
                        
                    # Yield event loop execution briefly to allow task to stream data
                    await asyncio.wait({task}, timeout=0.02)
                    
                # If task finished without timing out or being cancelled, return the raw MP3 bytes
                return await task
                
            except Exception:
                # If exhaust all retries or an explicit cancellation was tripped, raise the exception
                if attempt == self.config.tts_max_retries or cancel.is_set():
                    raise
            finally:
                # Cancel network stream task if cancelled or aborted early
                if not task.done():
                    task.cancel()
                # Await cancellation completion while suppressing CancelledError to keep logs clean
                await asyncio.gather(task, return_exceptions=True)
                
        return None

    def speak(self, text: str, *, cancel_event: threading.Event | None = None) -> bool:
        """
        Synthesize and play a sentence block to completion. 
        Block until playback finishes or is cancelled.
        """
        # Ensure a valid threading.Event exists even if caller did not supply one
        cancel = cancel_event if cancel_event is not None else threading.Event()

        # Track this cancellation token as the active one under lock
        with self._lock:
            if self._closed or cancel.is_set():
                return False
            self._current = cancel

        try:
            # Skip empty inputs
            if not text.strip():
                return False

            self.prepare()

            # Perform async network synthesis
            audio = asyncio.run(self._synthesize(text, cancel))
            if audio is None or cancel.is_set():
                return False

            # In-memory playback via pygame.mixer
            with BytesIO(audio) as source:
                # namehint='mp3' informs SDL_mixer of the file format without needing a temporary file on disk
                self._mixer.music.load(source, namehint='mp3')
                try:
                    # Recheck at device boundary; stop never clears this token.
                    if cancel.is_set():
                        return False

                    self._mixer.music.play()
                    deadline = time.monotonic() + self.config.playback_timeout

                    # Hardware playback polling loop
                    while self._mixer.music.get_busy():
                        if cancel.wait(0.02):
                            return False
                            
                        # Prevent hung audio drivers from freezing the pipeline
                        if time.monotonic() > deadline:
                            raise TimeoutError('Audio playback timeout')

                    # Return whether the playback stopped naturally or was cancelled
                    return not cancel.is_set()

                finally:
                    # Halt sound card generation and release memory buffers
                    try:
                        self._mixer.music.stop()
                    finally:
                        self._mixer.music.unload()

        finally:
            # Deregister active cancellation token under lock if it is still pointing to this operation
            with self._lock:
                if self._current is cancel:
                    self._current = None

    def stop(self) -> None:
        """Interrupt ongoing synthesis or playback immediately (barge-in entrypoint)."""
        with self._lock:
            if self._current is not None:
                # Tripping this event halts both the _synthesize asyncio loop and the playback polling loop
                self._current.set()

    def close(self) -> None:
        """Stop all active execution, unload hardware mixers, and shut down the service."""
        self.stop()
        with self._lock:
            self._closed = True
        if self._mixer_needs_close:
            self._mixer_needs_close = False
            self._prepared = False
            self._mixer.quit()
