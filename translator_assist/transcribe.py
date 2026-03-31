from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import whisper  # OpenAI Whisper
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError(
        "openai-whisper is required. Install it with 'pip install -U openai-whisper'."
    ) from exc

if not hasattr(whisper, "load_model"):
    raise RuntimeError(
        "The imported 'whisper' module is not OpenAI Whisper. "
        "Uninstall any 'whisper' package and install 'openai-whisper'."
    )


@dataclass
class Transcriber:
    model_name: str = "small"

    def __post_init__(self) -> None:
        # Device selection: GPU if available, else CPU. Whisper handles this internally via torch.
        self._model = whisper.load_model(self.model_name)

    def transcribe(self, audio: np.ndarray, source_lang: str) -> str:
        """Transcribe a pre-loaded waveform with Whisper, forcing the language.

        source_lang must be 'en' or 'es'.
        """
        if source_lang not in {"en", "es"}:
            raise ValueError(f"Unsupported source language: {source_lang}")

        # Disable language auto-detection by explicitly setting language.
        result: Any = self._model.transcribe(audio, language=source_lang, task="transcribe")
        text = result.get("text", "").strip()
        return text
