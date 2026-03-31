import argparse
import json
from pathlib import Path

from translator_assist.audio import load_audio, record_microphone, microphone_chunks
from translator_assist.transcribe import Transcriber
from translator_assist.translate import TranslatorBackend, BackendType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline EN↔ES transcription and translation assistant")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to input audio file (wav/mp3)")
    input_group.add_argument("--mic", action="store_true", help="Record from microphone once, then process")
    input_group.add_argument(
        "--mic-stream",
        action="store_true",
        help="Stream from microphone in chunks and print incremental results",
    )
    parser.add_argument("--source-lang", required=True, choices=["en", "es"], help="Source language of the audio")
    parser.add_argument("--backend", choices=["nllb", "opus"], default="nllb", help="Translation backend to use")
    parser.add_argument("--whisper-model", default="small", help="Whisper model size/name (default: small)")
    parser.add_argument(
        "--mic-duration",
        type=float,
        default=10.0,
        help="Duration in seconds for one-shot microphone recording (with --mic)",
    )
    parser.add_argument(
        "--mic-chunk-duration",
        type=float,
        default=5.0,
        help="Chunk duration in seconds for streaming microphone mode (with --mic-stream)",
    )
    # Mic/streaming could be added later as an option if desired
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transcriber = Transcriber(model_name=args.whisper_model)
    target_lang = "es" if args.source_lang == "en" else "en"
    backend_type = BackendType(args.backend)
    translator = TranslatorBackend(backend_type=backend_type, source_lang=args.source_lang, target_lang=target_lang)

    # File input mode
    if args.input:
        audio_path = Path(args.input)
        if not audio_path.is_file():
            raise SystemExit(f"Input file not found: {audio_path}")

        audio = load_audio(str(audio_path))
        transcription = transcriber.transcribe(audio, source_lang=args.source_lang)
        translation = translator.translate(transcription)

        result = {
            "transcription": transcription,
            "translation": translation,
            "source_lang": args.source_lang,
            "target_lang": target_lang,
        }
        print(json.dumps(result, ensure_ascii=False))
        return

    # One-shot microphone mode
    if args.mic:
        print(f"Recording microphone for {args.mic_duration} seconds at 16kHz mono...")
        audio = record_microphone(duration=args.mic_duration)
        transcription = transcriber.transcribe(audio, source_lang=args.source_lang)
        translation = translator.translate(transcription)

        result = {
            "transcription": transcription,
            "translation": translation,
            "source_lang": args.source_lang,
            "target_lang": target_lang,
        }
        print(json.dumps(result, ensure_ascii=False))
        return

    # Streaming microphone mode (chunked)
    if args.mic_stream:
        print(
            "Streaming from microphone. Press Ctrl+C to stop. "
            f"Chunk duration: {args.mic_chunk_duration} seconds."
        )
        try:
            for i, chunk in enumerate(microphone_chunks(chunk_duration=args.mic_chunk_duration), start=1):
                transcription = transcriber.transcribe(chunk, source_lang=args.source_lang)
                translation = translator.translate(transcription)
                result = {
                    "chunk_index": i,
                    "transcription": transcription,
                    "translation": translation,
                    "source_lang": args.source_lang,
                    "target_lang": target_lang,
                }
                print(json.dumps(result, ensure_ascii=False))
        except KeyboardInterrupt:
            print("\nStopped microphone streaming.")
        return


if __name__ == "__main__":
    main()
