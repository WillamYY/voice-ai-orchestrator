"""Validated runtime configuration; .env is resolved from the project root."""
from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', env_file_encoding='utf-8',
                                     case_sensitive=False, extra='forbid', hide_input_in_errors=True)
    # Descriptions are also the source for .env.example and docs/configuration.md.
    # Regenerate them with: python scripts/sync_settings_docs.py
    use_gpu: bool = Field(default=True, description='Prefer CUDA for local ASR/LLM when available; otherwise fall back to CPU. Does not install CUDA.')
    base_dir: str = Field(default=str(ROOT), description='Root for resolving relative model/cache paths. Defaults to the project root; use an absolute path when overriding. Does not relocate .env.')

    sample_rate: int = Field(default=16000, description='Mono PCM16 capture rate in Hz. This ASR adapter requires exactly 16000.')
    chunk_duration: float = Field(default=0.02, description='Capture/VAD frame duration in seconds: 0.01, 0.02 or 0.03 only. At 16000 Hz, 0.02 means 320 samples / 640 bytes.')
    silence_threshold: float = Field(default=0.8, gt=0, le=10, description='Consecutive non-speech seconds required to end an utterance. Larger values delay the reply; rounded up to whole frames.')
    vad_mode: int = Field(default=3, ge=0, le=3, description='WebRTC VAD aggressiveness: 0 least restrictive, 3 most restrictive speech detection. Classifies frames; does not remove noise or echo.')
    speech_start_ms: int = Field(default=100, ge=20, le=500, description='Consecutive voiced milliseconds needed to confirm onset, rounded up to whole frames. Buffer retains these frames plus up to 200 ms of preceding audio.')
    max_utterance_seconds: float = Field(default=30, ge=1, le=120, description='Force an endpoint at this many seconds of emitted audio, including pre-roll and trailing silence. Rounded up to whole frames; must exceed the speech-start window.')
    input_device_index: int | None = Field(default=None, description='PyAudio input device index from verify_env.py --audio. Omit to use the system default; an empty value is not a valid integer.')

    asr_model: str = Field(default='paraformer-zh-streaming', description='Compatible FunASR streaming model alias or local model directory. An alias may download weights; the default model is Chinese-oriented.')
    asr_cache_dir: str = Field(default='.cache/modelscope', description='ModelScope download cache. An existing MODELSCOPE_CACHE environment variable takes precedence. Relative paths resolve under BASE_DIR.')
    llm_backend: Literal['local', 'api'] = Field(default='api', description='LLM provider mode: api uses the configured compatible chat-completions endpoint; local loads existing weights.')
    llm_model_path: str = Field(default='', description='Local tokenizer/model directory; set this when LLM_BACKEND=local. Files must already exist. Relative paths resolve under BASE_DIR. Unused in API mode.')
    llm_api_url: str = Field(default='http://localhost:11434/v1', description='HTTP(S) API base URL, including the server API prefix if required. Used only in api mode.')
    llm_api_model: str = Field(default='llm0', description='Exact model identifier exposed by the API server; used only in api mode.')
    llm_api_key: str = Field(default='NA', repr=False, description='API credential; NA is a placeholder for servers without authentication. Keep real credentials only in private environment configuration.')
    llm_max_tokens: int = Field(default=256, ge=1, le=4096, description='Maximum generated tokens per response, passed to local/API generation. Tokens are not characters.')
    llm_input_tokens: int = Field(default=2048, ge=128, le=32768, description='Local prompt token budget, further limited by model context minus output budget. Drops oldest turns, retaining system/current input. Not applied to API requests.')
    llm_timeout: float = Field(default=90, gt=0, description='Maximum inference idle seconds before first text, between emitted text chunks, or before generation finishes, after provider preparation. Full-output-queue waits use DOWNSTREAM_TIMEOUT instead. Completed output can drain without this timeout; native cancellation remains cooperative.')
    downstream_timeout: float = Field(default=300, gt=0, description='Maximum seconds continuously blocked on a queue handoff, or waiting for playback completion without a completed segment. Separate from inference idle and per-segment TTS/playback timeouts; cancellation remains cooperative.')
    request_timeout: float = Field(default=15, gt=0, description='API client HTTP timeout in seconds, separate from LLM_TIMEOUT; SDK retries are disabled. Used only in api mode.')
    history_turns: int = Field(default=15, ge=1, le=100, description='Number of recent user-led turns retained after each response. Only completed assistant playback segments enter history; interrupted segments are omitted.')
    system_prompt: str = Field(default='You are a helpful voice assistant. Reply concisely in the language of the user.', description='System instruction for each LLM request. Local placement is selected automatically from the model template; API mode keeps the system role. Empty or whitespace-only text disables it. prompt.txt is not loaded automatically.')

    tts_model: str = Field(default='zh-CN-XiaoyiNeural', description='Edge TTS voice name. Synthesis requires internet and sends the filtered assistant text to the service.')
    tts_max_retries: int = Field(default=1, ge=0, le=3, description='Additional synthesis attempts before playback; total attempts = 1 + this value. No retry after playback starts.')
    tts_timeout: float = Field(default=30, gt=0, description='Synthesis timeout in seconds per attempt, per text segment. Retries can multiply total wait time.')
    tts_max_segment_bytes: int = Field(default=16 * 1024 * 1024, gt=0, description='Maximum compressed MP3 bytes accumulated for one segment, checked before extending the buffer. Default 16777216 = 16 MiB; not a total-process memory cap.')
    playback_timeout: float = Field(default=120, gt=0, description='Maximum playback seconds per synthesized segment; exceeds this only by polling/driver delays.')
    segment_chars: int = Field(default=120, ge=20, le=500, description='Target maximum raw text characters per segment before cleanup. Split at sentence delimiters or a whitespace/hard length boundary; no timed flush.')

    text_filter_enabled: bool = Field(default=True, description='Enable speech text cleanup inside the segmenter after each segment is extracted, including the final tail. False preserves raw text; empty cleaned segments are skipped.')
    text_filter_markdown: bool = Field(default=True, description='When filtering is enabled, remove the literal symbols *, _, backtick, ~, # and >. Lightweight cleanup, not a Markdown parser; may alter code or math.')
    text_filter_emoji: bool = Field(default=True, description='When filtering is enabled, remove common emoji ranges, flags, modifiers, joiners, variation selectors and keycaps. Heuristic: not all Unicode emoji; some non-emoji symbols also match.')
    text_filter_emoticons: bool = Field(default=True, description='When filtering is enabled, remove the common ASCII faces listed in SpeechTextFilter.EMOTICONS, including :) and :D. Case-sensitive substring matching.')
    text_filter_remove: list[str] = Field(default_factory=list, description='Extra literal words/characters to delete within each completed segment, as a JSON array of nonempty strings. Case-sensitive substrings, longest match first. LLM chunk splits are fine within a segment; matches across segment boundaries are not removed. No regex or word-boundary matching.')

    audio_queue_frames: int = Field(default=250, ge=10, le=5000, description='Capture-to-ASR queue capacity in packets (audio frames plus endpoint markers). Default holds about 5 seconds at 20 ms/frame. Overflow is fatal rather than dropping speech.')
    text_queue_size: int = Field(default=8, ge=1, le=100, description='Capacity of each text queue: completed user turns and TTS segments/end markers. Producers wait when full, respecting cancellation.')
    shutdown_timeout: float = Field(default=5, gt=0, le=30, description='Manager worker-join deadline in seconds; also used separately for inference worker joins. Native calls may outlive it and are reported. Not a strict total process-exit deadline.')
    barge_in: bool = Field(default=True, description='Cancel the current assistant response on a new nonempty completed user transcript. Explicit stop commands work even when False.')
    stop_phrases: list[str] = Field(default_factory=lambda: ['先暂停一下', 'Stop it please'], description='Stop phrases as a JSON array. Case-insensitive literal substring matching on completed transcripts; surrounding phrase whitespace is ignored, internal spaces/punctuation are preserved. A match consumes the entire utterance and cancels active/pending replies even with BARGE_IN=False. Choose distinctive longer phrases; blank entries are invalid, [] disables verbal stop commands. Does not bypass VAD/ASR endpoint delay.')
    barge_in_on_speech_start: bool = Field(default=False, description='With BARGE_IN=True, also cancel on VAD-confirmed onset, before ASR text. Faster but sensitive to noise/echo; not acoustic echo cancellation.')

    @model_validator(mode='after')
    def validate_paths(self):
        if any(not phrase.strip() for phrase in self.stop_phrases):
            raise ValueError('STOP_PHRASES entries must not be blank')
        if any(not item for item in self.text_filter_remove):
            raise ValueError('TEXT_FILTER_REMOVE entries must be nonempty strings')
        if self.sample_rate != 16000:
            raise ValueError('SAMPLE_RATE must be 16000 for this ASR adapter')
        if self.chunk_duration not in (0.01, 0.02, 0.03):
            raise ValueError('CHUNK_DURATION must be 0.01, 0.02 or 0.03 for WebRTC VAD')
        for name in ('asr_cache_dir', 'llm_model_path'):
            if name == 'llm_model_path' and not self.llm_model_path.strip():
                continue
            path = Path(getattr(self, name))
            setattr(self, name, str(path if path.is_absolute() else Path(self.base_dir) / path))
        if self.llm_backend == 'api' and not self.llm_api_url.startswith(('http://', 'https://')):
            raise ValueError('LLM_API_URL must be an HTTP(S) URL in API mode')
        if self.speech_start_ms / 1000 >= self.max_utterance_seconds:
            raise ValueError('Speech start window must be shorter than maximum utterance')
        return self


class DefaultSettings(Settings):
    """Validated defaults and explicit overrides, without external configuration."""

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings,
                                   dotenv_settings, file_secret_settings):
        return (init_settings,)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache runtime configuration on first use, never on schema import."""
    return Settings()
