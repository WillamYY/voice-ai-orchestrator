"""Messages and response lifetime shared across workers."""
from dataclasses import dataclass, field
import threading


@dataclass(frozen=True)
class SpeechStarted:
    pass


@dataclass(frozen=True)
class AudioPacket:
    utterance_id: int
    data: bytes | None  # None is end of this utterance (silence), never shutdown.


@dataclass(frozen=True)
class UserTurn:
    utterance_id: int
    text: str
    generation: int = 0  # Explicit stop invalidates work from older generations.


@dataclass
class Response:
    response_id: int
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    played: list[str] = field(default_factory=list)
    error: str | None = None
    generation: int = 0


@dataclass(frozen=True)
class SpeechSegment:
    response: Response
    text: str | None  # None acknowledges end of response after earlier playback.
