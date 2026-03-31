# translator assist

Offline EN↔ES transcription + translation pipeline using Whisper and NLLB/OPUS.

## Setup

Create a virtual environment and install dependencies:

```bash
cd translatorAssist
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# If Whisper was installed incorrectly earlier, you may need:
# pip uninstall -y whisper
# pip install -U openai-whisper
```

The first run will download Whisper, NLLB, or Marian models; after that, everything runs locally with no external APIs.

## Usage

### CLI (file input)

Basic file-based CLI usage:

```bash
python main.py --input path/to/audio.wav --source-lang en --backend nllb
```

Arguments:

- `--input <file>`: Path to audio file (wav/mp3 supported by librosa).
- `--source-lang en|es`: Source language of the spoken audio.
- `--backend nllb|opus`: Translation backend. `nllb` (default) for higher quality, `opus` for lighter models.
- `--whisper-model`: Whisper model size/name (default: `small`).

The program prints structured JSON to stdout, for example:

```json
{
  "transcription": "...",
  "translation": "...",
  "source_lang": "en",
  "target_lang": "es"
}
```

### CLI (microphone)

One-shot microphone recording (record, then process once):

```bash
python main.py --mic --source-lang en --backend nllb --mic-duration 10
```

Streaming-style microphone mode (process audio in chunks and print incremental results):

```bash
python main.py --mic-stream --source-lang en --backend nllb --mic-chunk-duration 5
```

Both modes run fully offline; streaming prints one JSON object per chunk.

### REST API (FastAPI)

You can also run a REST API using FastAPI:

```bash
uvicorn translator_assist.api:app --host 0.0.0.0 --port 8000
```

Example request with `curl`:

```bash
curl -X POST "http://localhost:8000/transcribe-translate?source_lang=en&backend=nllb&whisper_model=small" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/audio.wav"
```

The response JSON has the same structure as the CLI output:

```json
{
  "transcription": "...",
  "translation": "...",
  "source_lang": "en",
  "target_lang": "es"
}
```

### Desktop GUI

You can also use a simple desktop GUI for microphone streaming.

Run:

```bash
python gui.py
```

The window provides:

- A source-language switch (English or Spanish input).
- A **Start Mic** button to begin streaming from the default microphone.
- A **Stop Mic** button to end streaming.
- A large text area that appends one block per chunk, with transcription and translation side by side, separated by blank lines between chunks.

## Notes

- Audio is normalized to 16kHz mono before transcription.
- Whisper language auto-detection is disabled; it is forced to the provided `--source-lang`.
- GPU will be used automatically by PyTorch/Whisper if available; otherwise CPU is used.
- Microphone streaming and REST API (FastAPI) can be added later if needed.
