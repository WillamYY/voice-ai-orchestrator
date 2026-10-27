"""
Structural provider contracts; implementations do not inherit from these.

Protocols describe the manager's dependencies for static type checkers. They
provide no default behavior or runtime validation. Each provider owns startup
and cleanup, including safe repeated close/stop calls.
"""
from collections.abc import Iterator
from threading import Event
from typing import Protocol, Self

from src.core.events import SpeechStarted


class RecorderProvider(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> None: ...

    def stream(self) -> Iterator[SpeechStarted | bytes | None]:
        """Yield speech starts, audio chunks, and None utterance endpoints."""
        ...

    def stop(self) -> None: ...


class ASRProvider(Protocol):
    def prepare(self) -> None: ...

    def transcribe(self, audio_chunk: bytes) -> str:
        """Partial text does not commit a turn."""
        ...

    def finish_utterance(self) -> str:
        """Return a complete transcript once and reset."""
        ...

    def reset_state(self) -> None:
        """Discard audio/transcript without inference."""
        ...

    def close(self) -> None: ...


class LLMProvider(Protocol):
    def prepare(self) -> None: ...

    def generate_response(
        self, user_input: str, history: list[dict[str, str]], *,
        cancel_event: Event | None = None,
    ) -> Iterator[str]:
        """
        Stream text using prior history plus current input exactly once.
        Raise provider failures; never speak exceptions as assistant content.
        """
        ...

    def close(self) -> None: ...


class TTSProvider(Protocol):
    def prepare(self) -> None: ...

    def speak(self, text: str, *, cancel_event: Event | None = None) -> bool:
        """True only when the segment completed playback; False if cancelled."""
        ...

    def stop(self) -> None:
        """Cancel active work without reviving any previous response."""
        ...

    def close(self) -> None: ...
