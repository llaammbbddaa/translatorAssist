from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import MarianMTModel, MarianTokenizer


class BackendType(str, Enum):
    NLLB = "nllb"
    OPUS = "opus"


@lru_cache(maxsize=4)
def _load_model_and_tokenizer(model_name: str):
    """Load tokenizer/model pair for non-Marian seq2seq models.

    Marian/OPUS models are handled explicitly via MarianMTModel/MarianTokenizer
    because some transformers versions do not map MarianConfig correctly in
    AutoTokenizer.from_pretrained.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


@dataclass
class TranslatorBackend:
    backend_type: BackendType
    source_lang: str
    target_lang: str
    _model_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.source_lang == self.target_lang:
            raise ValueError("source_lang and target_lang must differ")

        if self.backend_type == BackendType.NLLB:
            # NLLB: single multilingual model; select language codes per direction.
            self._model_name = self._model_name or "facebook/nllb-200-distilled-600M"
            self._src_code = "eng_Latn" if self.source_lang == "en" else "spa_Latn"
            self._tgt_code = "spa_Latn" if self.target_lang == "es" else "eng_Latn"

            # For NLLB, configure tokenizer with src_lang as recommended in HF docs.
            # We keep a dedicated loading path here instead of using _load_model_and_tokenizer
            # so that src_lang is always set correctly for encoding.
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, src_lang=self._src_code)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        elif self.backend_type == BackendType.OPUS:
            # Marian/OPUS: direction-specific models.
            if self.source_lang == "en" and self.target_lang == "es":
                self._model_name = self._model_name or "Helsinki-NLP/opus-mt-en-es"
            elif self.source_lang == "es" and self.target_lang == "en":
                self._model_name = self._model_name or "Helsinki-NLP/opus-mt-es-en"
            else:
                raise ValueError("Unsupported language pair for OPUS backend")
            self._src_code = None
            self._tgt_code = None

            # For Marian models, use the dedicated tokenizer/model classes to
            # avoid AutoTokenizer configuration issues across transformers versions.
            self._tokenizer = MarianTokenizer.from_pretrained(self._model_name)
            self._model = MarianMTModel.from_pretrained(self._model_name)
            return
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}")

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""

        tokenizer = self._tokenizer
        model = self._model

        if self.backend_type == BackendType.NLLB:
            tokenizer.src_lang = self._src_code  # type: ignore[attr-defined]
            encoded = tokenizer(text, return_tensors="pt")

            # Newer versions of transformers expose `lang_code_to_id`; for others,
            # fall back to constructing the BOS language token (e.g. "<2spa_Latn>").
            lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
            if isinstance(lang_code_to_id, dict):
                forced_bos_token_id = lang_code_to_id[self._tgt_code]
            else:
                lang_token = f"<2{self._tgt_code}>"
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(lang_token)

            generated_tokens = model.generate(**encoded, forced_bos_token_id=forced_bos_token_id)
            out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return out.strip()

        # OPUS / Marian
        encoded = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(**encoded)
        out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return out.strip()
