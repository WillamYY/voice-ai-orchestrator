# Voice AI Orchestrator

A Windows x64 Python prototype for continuous microphone capture, streaming speech recognition, LLM responses, and sentence-level speech playback. Capture, recognition, generation, and playback run in separate workers. The supported setup is one local session. Use headphones: speaker echo cancellation is not implemented.

## Requirements

The dependency lock and supported setup target Windows x64. Other operating systems are not currently covered by the lockfile or project checks.

- Windows x64 and Python 3.12 (the project metadata also accepts Python 3.11)
- [uv](https://docs.astral.sh/uv/)
- A working microphone and speaker, with the audio drivers required by PyAudio and pygame
- Internet access for Edge TTS and, on first use, the default ASR model download
- Local LLM weights, or access to a compatible OpenAI chat-completions API

## Install

Run these commands from the project root in PowerShell:

```powershell
uv sync --locked
Copy-Item .env.example .env   # Skip if .env already exists
uv run --locked python main.py --smoke-test
```

The smoke test uses diagnostic providers included in `src/diagnostics.py`; it does not require model files, network access, or audio devices. Torch and torchaudio use the CUDA 12.8 package index. Downloads are large; a compatible NVIDIA GPU and driver are required for CUDA. With `USE_GPU=True`, the app falls back to CPU when CUDA is unavailable.

## Configure models

The default LLM backend is API. Configure `LLM_API_URL`, `LLM_API_MODEL`, and `LLM_API_KEY` in `.env` for your compatible chat-completions server. For a server without authentication, the default API key value is `NA`.

For local inference, set `LLM_BACKEND=local` and `LLM_MODEL_PATH` to an existing directory containing model weights and matching tokenizer files. The app adapts system instructions and interrupted conversations to the tokenizer's chat template before loading LLM weights or opening the microphone. Model files are not bundled or downloaded automatically.

The default ASR model, `paraformer-zh-streaming`, downloads its weights on first use. The default cache of ASR model directory is `.cache/modelscope`. Set `ASR_MODEL` to a compatible local streaming model directory for offline use with other possible use cases. ASR requires mono 16 kHz PCM audio. Edge TTS requires internet access and sends assistant text to its service.

## Run

```powershell
# Check installed packages and configured local LLM files
uv run --locked python verify_env.py

# Check audio device availability without recording or playback
uv run --locked python verify_env.py --audio

# Start microphone conversation
uv run --locked python main.py

# Type one prompt, use the configured LLM, and print the answer
uv run --locked python main.py --text "Say hello in one short sentence." --no-audio
```

When the app reports `Ready`, speak a short question and pause. Press Ctrl+C to exit. `SILENCE_THRESHOLD` sets the consecutive non-speech time used to end an utterance; its default is 0.8 seconds. `--no-audio` prints responses and acknowledges them in conversation history; it does not test speaker playback.

Optional checks can load models or contact the configured API:

```powershell
# Load configured models; ASR may download model weights
uv run --locked python verify_env.py --load-models

# Check the configured API model list when LLM_BACKEND=api
uv run --locked python verify_env.py --network
```

## Interruption

By default, a new nonempty completed transcript interrupts the active response (`BARGE_IN=True`). Set `BARGE_IN=False` to disable ordinary interruption. You can also enable VAD-onset interruption with `BARGE_IN_ON_SPEECH_START=True`; noise or echo may trigger it.

Explicit stop phrases work regardless of `BARGE_IN`. Set distinctive phrases in `.env` as a JSON array:

```dotenv
STOP_PHRASES='["先暂停一下", "停一下","Stop it please"]'
```

Matching is a case-insensitive literal substring search on the completed transcript. Internal spaces and punctuation must match; surrounding phrase whitespace is ignored. The whole matching utterance is consumed. Set `STOP_PHRASES=[]` to disable verbal stop commands. Recognition waits for speech endpointing and ASR, so background noise or a full input queue can delay or prevent a match.

## Configuration and limits

Copy `.env.example` to `.env` for local overrides. Configuration precedence is constructor overrides, process environment, `.env`, then defaults. Setting names are case-insensitive; unknown `.env` keys and invalid values are rejected. Relative model and cache paths resolve under `BASE_DIR`, which defaults to the project root.

ASR runs locally. API LLM mode sends user text and conversation context to the configured API. Edge TTS sends assistant text to its service. Logs contain recognized transcripts and completed assistant text. Audio playback is held in memory; the app does not write microphone recordings or speech files.

This is a personal project as a source prototype, not a production service. It has no echo cancellation or speaker identification, runs one session, and plays complete synthesized segments rather than incremental audio. Recognition accuracy, interruption delay, noise handling, and model memory use depend on the device and require local evaluation. The default ASR model is intended for Chinese speech input. Other languages are not currently supported by the default streaming ASR pipeline. Microphone/audio testing are currently performed on Windows. Linux may work, but has not been verified.