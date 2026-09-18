from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from config import WhisperConfig


@lru_cache(maxsize=1)
def _load_model() -> WhisperModel:
    print(
        f"[transcriber] loading whisper model '{WhisperConfig.model}' "
        f"(device={WhisperConfig.device}, compute={WhisperConfig.compute_type})"
    )
    return WhisperModel(
        WhisperConfig.model,
        device=WhisperConfig.device,
        compute_type=WhisperConfig.compute_type,
    )


def _average_confidence(segments: list) -> float:
    log_probs = [s.avg_logprob for s in segments if s.avg_logprob is not None]
    if not log_probs:
        return 0.0
    avg = sum(log_probs) / len(log_probs)
    return round(min(max((avg + 1.0), 0.0), 1.0), 4)


def transcribe_file(audio_path: str | Path) -> tuple[str, float]:
    model = _load_model()
    segments, _info = model.transcribe(str(audio_path), beam_size=5, vad_filter=True)
    segments = list(segments)
    transcript = " ".join(s.text.strip() for s in segments).strip()
    confidence = _average_confidence(segments)
    return transcript, confidence


def transcribe_bytes(audio_bytes: bytes) -> tuple[str, float]:
    model = _load_model()
    audio_buffer = io.BytesIO(audio_bytes)
    segments, _info = model.transcribe(audio_buffer, beam_size=5, vad_filter=True)
    segments = list(segments)
    transcript = " ".join(s.text.strip() for s in segments).strip()
    confidence = _average_confidence(segments)
    return transcript, confidence
