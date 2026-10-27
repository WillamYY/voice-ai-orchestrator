"""Read-only installation checks. Optional device/network/model checks are explicit."""
import argparse
import importlib
import os
from importlib.metadata import version
from pathlib import Path
import sys


PACKAGES = {
    'numpy': 'numpy', 'torch': 'torch', 'torchaudio': 'torchaudio',
    'funasr': 'funasr', 'modelscope': 'modelscope', 'soundfile': 'soundfile',
    'pyaudio': 'PyAudio', 'webrtcvad': 'webrtcvad-wheels', 'pygame': 'pygame',
    'edge_tts': 'edge-tts', 'pydantic_settings': 'pydantic-settings',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', action='store_true', help='list input devices and check selected input/output; no recording')
    parser.add_argument('--network', action='store_true', help='query configured API model list when in API mode')
    parser.add_argument('--load-models', action='store_true', help='load configured models; ASR alias may download weights')
    args = parser.parse_args(argv)
    failures = []

    def check(label, action):
        try:
            value = action()
            print(f'[OK] {label}: {value if value is not None else "available"}', flush=True)
        except Exception as error:
            failures.append(label)
            print(f'[FAIL] {label}: {error}', flush=True)

    print(f'Python {sys.version.split()[0]} at {sys.executable}', flush=True)
    try:
        from src.config.settings import get_settings
        settings = get_settings()
    except Exception as error:
        print(f'[FAIL] Configuration: {error}')
        return 1
    print(f'LLM backend: {settings.llm_backend}; ASR: {settings.asr_model}', flush=True)
    os.environ.setdefault('MODELSCOPE_CACHE', settings.asr_cache_dir)
    packages = dict(PACKAGES)
    packages['transformers' if settings.llm_backend == 'local' else 'openai'] = (
        'transformers' if settings.llm_backend == 'local' else 'openai')
    for module, distribution in packages.items():
        def load(module=module, distribution=distribution):
            importlib.import_module(module)
            return version(distribution)
        check(module, load)
    if settings.llm_backend == 'local':
        def local_files():
            root = Path(settings.llm_model_path)
            required = ['config.json', 'tokenizer_config.json']
            if any(not (root / name).is_file() for name in required):
                raise FileNotFoundError(f'Missing model/tokenizer configuration in {root}')
            if not list(root.glob('*.safetensors')) and not list(root.glob('pytorch_model*.bin')):
                raise FileNotFoundError(f'Missing local weights in {root}')
            return root
        check('Local LLM files', local_files)
    if 'torch' not in failures:
        import torch
        if settings.use_gpu and not torch.cuda.is_available():
            print('[WARN] CUDA unavailable: both ASR and LLM will use CPU. Expect slower responses.')
        else:
            print('[OK] Inference device: ' + ('cuda' if settings.use_gpu else 'cpu'))
    if args.audio:
        def audio_devices():
            import pyaudio
            pa = pyaudio.PyAudio()
            try:
                for i in range(pa.get_device_count()):
                    item = pa.get_device_info_by_index(i)
                    if item['maxInputChannels']:
                        print(f'  Input {i}: {item["name"]}')
                index = settings.input_device_index
                if index is None:
                    index = int(pa.get_default_input_device_info()['index'])
                pa.is_format_supported(settings.sample_rate, input_device=index,
                                       input_channels=1, input_format=pyaudio.paInt16)
                return f'16 kHz mono PCM16 supported on input {index}'
            finally:
                pa.terminate()
        check('Microphone format', audio_devices)
        def output_device():
            import pygame
            pygame.mixer.init()
            pygame.mixer.quit()
            return 'output mixer initialized and closed without playback'
        check('Speaker output', output_device)
    if args.network and settings.llm_backend == 'api':
        def api_models():
            from openai import OpenAI
            with OpenAI(base_url=settings.llm_api_url, api_key=settings.llm_api_key,
                        timeout=settings.request_timeout, max_retries=0) as client:
                ids = [m.id for m in client.models.list().data]
                if settings.llm_api_model not in ids:
                    raise ValueError(f'Configured model {settings.llm_api_model!r} not in {ids}')
                return settings.llm_api_model
        check('API model availability', api_models)
    if args.load_models:
        from src.services.asr_service import FunASRService
        from src.services.llm_service import LLMService
        for label, service in [('ASR loading', FunASRService(config=settings)),
                               ('LLM loading', LLMService(config=settings))]:
            try:
                check(label, service.prepare)
            finally:
                service.close()
    print('Edge TTS requires internet; synthesis/playback is tested with python main.py --text "Hello".')
    if not args.load_models:
        print('Model loading/ASR downloads were not tested. Use --load-models for that check.')
    print(f'Checks complete: {len(failures)} failure(s).')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
