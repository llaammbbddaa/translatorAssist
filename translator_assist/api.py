from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

from .audio import TARGET_SR
from .transcribe import Transcriber
from .translate import TranslatorBackend, BackendType


app = FastAPI(title="Translator Assist API", version="0.1.0")


_transcriber_cache: Optional[Transcriber] = None


def get_transcriber(model_name: str) -> Transcriber:
    global _transcriber_cache
    if _transcriber_cache is None or _transcriber_cache.model_name != model_name:
        _transcriber_cache = Transcriber(model_name=model_name)
    return _transcriber_cache


def get_translator(source_lang: str, target_lang: str, backend: str) -> TranslatorBackend:
    backend_type = BackendType(backend)
    return TranslatorBackend(backend_type=backend_type, source_lang=source_lang, target_lang=target_lang)


async def _load_audio_from_upload(file: UploadFile) -> np.ndarray:
    import io
    import soundfile as sf

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        with io.BytesIO(data) as buf:
            audio, sr = sf.read(buf, dtype="float32")
    except Exception as exc:  # pragma: no cover - depends on codecs
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {exc}") from exc

    if audio.ndim > 1:
        # Convert to mono
        audio = audio.mean(axis=1)

    if sr != TARGET_SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    return audio.astype("float32")


@app.post("/transcribe-translate")
async def transcribe_and_translate(
    file: UploadFile = File(..., description="Audio file (wav/mp3)"),
    source_lang: str = Query(..., description="Source language: en or es"),
    backend: str = Query("nllb", description="Translation backend: nllb or opus"),
    whisper_model: str = Query("small", description="Whisper model size/name"),
):
    if source_lang not in {"en", "es"}:
        raise HTTPException(status_code=400, detail="source_lang must be 'en' or 'es'")

    audio = await _load_audio_from_upload(file)

    transcriber = get_transcriber(whisper_model)
    transcription = transcriber.transcribe(audio, source_lang=source_lang)

    target_lang = "es" if source_lang == "en" else "en"
    translator = get_translator(source_lang=source_lang, target_lang=target_lang, backend=backend)
    translation = translator.translate(transcription)

    result = {
        "transcription": transcription,
        "translation": translation,
        "source_lang": source_lang,
        "target_lang": target_lang,
    }
    return JSONResponse(content=result)


@app.get("/")
async def root() -> dict:
    return {
        "message": "Translator Assist API",
        "endpoints": {
            "POST /transcribe-translate": "Upload audio and get transcription+translation",
        },
    }
