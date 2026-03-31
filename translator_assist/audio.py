from typing import Generator

import librosa
import numpy as np

TARGET_SR = 16000


def load_audio(path: str) -> np.ndarray:
    """Load an audio file and normalize to 16kHz mono float32 waveform.

    Returns a 1D numpy array suitable for Whisper.
    """
    audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    # librosa already returns float32 in [-1, 1]
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    return audio.astype("float32")


def record_microphone(duration: float) -> np.ndarray:
    """Record from the default microphone for a fixed duration in seconds.

    Returns a 1D float32 numpy array at 16kHz mono.
    """
    import sounddevice as sd

    frames = int(TARGET_SR * duration)
    audio = sd.rec(frames, samplerate=TARGET_SR, channels=1, dtype="float32")
    sd.wait()
    # sd.rec returns shape (frames, channels)
    return np.squeeze(audio, axis=1).astype("float32")


def microphone_chunks(chunk_duration: float = 5.0) -> Generator[np.ndarray, None, None]:
    """Yield successive chunks of microphone audio as 1D float32 arrays.

    This is a simple building block for streaming-style processing. Each
    yielded chunk has length ``chunk_duration`` seconds at 16kHz mono.
    """
    import sounddevice as sd

    frames_per_chunk = int(TARGET_SR * chunk_duration)
    with sd.InputStream(samplerate=TARGET_SR, channels=1, dtype="float32") as stream:
        while True:
            chunk, overflowed = stream.read(frames_per_chunk)
            # chunk: (frames, channels)
            mono = np.squeeze(chunk, axis=1).astype("float32")
            yield mono
