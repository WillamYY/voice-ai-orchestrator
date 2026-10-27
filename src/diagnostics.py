"""Finite text sources and fake providers for smoke tests and typed diagnostics."""
from collections.abc import Iterator
from threading import Event
from typing import Self

from src.core.events import SpeechStarted


class TextRecorder:
    def __init__(self, texts):
        self.texts = texts
        self.stopped = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def stop(self) -> None:
        self.stopped = True

    def stream(self) -> Iterator[SpeechStarted | bytes | None]:
        for text in self.texts:
            if self.stopped:
                break
            yield SpeechStarted()
            yield text.encode('utf-8')
            yield None


class TextASR:
    """Test-only bytes-to-text adapter; never used for microphone PCM."""
    def __init__(self):
        self.reset_state()

    def prepare(self) -> None:
        pass

    def transcribe(self, audio_chunk: bytes) -> str:
        self.buffer.extend(audio_chunk)
        return ''

    def finish_utterance(self) -> str:
        text = self.buffer.decode('utf-8')
        self.reset_state()
        return text

    def reset_state(self) -> None:
        self.buffer = bytearray()

    def close(self) -> None:
        self.reset_state()


class EchoLLM:
    def prepare(self) -> None:
        pass

    def generate_response(
        self, user_input: str, history: list[dict[str, str]], *,
        cancel_event: Event | None = None,
    ) -> Iterator[str]:
        for part in ('Received: ', user_input, '.'):
            if cancel_event is not None and cancel_event.is_set():
                return
            yield part

    def close(self) -> None:
        pass


class ConsoleTTS:
    """Acknowledge printed text in diagnostics; this does not validate audio."""
    def prepare(self) -> None:
        pass

    def speak(self, text: str, *, cancel_event: Event | None = None) -> bool:
        if cancel_event is not None and cancel_event.is_set():
            return False
        print(f'[text output] {text.strip()}', flush=True)
        return True

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.stop()
