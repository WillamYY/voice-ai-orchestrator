"""Canonical CLI: live microphone, typed input, or hardware-free smoke test."""
import argparse
import logging
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description='Voice AI Orchestrator (use headphones for barge-in)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke-test', action='store_true', help='fake providers, no models/network/audio')
    mode.add_argument('--text', help='one typed utterance through the real LLM and TTS, bypassing ASR')
    parser.add_argument('--no-audio', action='store_true', help='print output instead of using TTS')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    try:
        from src.config.settings import DefaultSettings, get_settings
        from src.core.manager import ConversationManager
        from src.diagnostics import TextRecorder, TextASR, EchoLLM, ConsoleTTS
        if args.smoke_test:
            config = DefaultSettings(barge_in=False, use_gpu=False, system_prompt='')
            app = ConversationManager(config=config, recorder=TextRecorder(['hello', 'second turn']),
                                      asr=TextASR(), llm=EchoLLM(), tts=ConsoleTTS())
        else:
            kwargs = {'config': get_settings()}
            if args.text is not None:
                if not args.text.strip():
                    parser.error('--text must not be blank')
                kwargs.update(recorder=TextRecorder([args.text]), asr=TextASR())
            if args.no_audio:
                kwargs['tts'] = ConsoleTTS()
            app = ConversationManager(**kwargs)
        app.start()
        if args.smoke_test:
            expected = ['user', 'assistant', 'user', 'assistant']
            if [m['role'] for m in app.history] != expected or app.errors:
                raise RuntimeError('Smoke test did not complete both turns')
            print('SMOKE TEST PASSED: two turns, playback acknowledgments, clean shutdown.')
        if app.errors:
            logging.error('Session finished with %d error(s). See log above.', len(app.errors))
            return 1
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        logging.error('Cannot complete session: %s', error, exc_info=True)
        logging.error('Run python verify_env.py to check setup.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
