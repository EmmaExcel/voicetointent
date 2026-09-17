from __future__ import annotations

import io
import wave

import numpy as np

from core.transcriber import _average_confidence, transcribe_bytes, transcribe_file


def _make_silent_wav(duration_seconds: float = 0.5, sample_rate: int = 16000) -> bytes:
    num_samples = int(sample_rate * duration_seconds)
    audio = np.zeros(num_samples, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def test_average_confidence_empty():
    assert _average_confidence([]) == 0.0


def test_average_confidence_clamps_to_range():
    segment = type("S", (), {"avg_logprob": -5.0})()
    result = _average_confidence([segment])
    assert 0.0 <= result <= 1.0


def test_average_confidence_normal():
    segment = type("S", (), {"avg_logprob": -0.3})()
    result = _average_confidence([segment])
    assert 0.0 <= result <= 1.0


def test_transcribe_bytes_returns_tuple(silent_wav_bytes):
    transcript, confidence = transcribe_bytes(silent_wav_bytes)
    assert isinstance(transcript, str)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


def test_transcribe_bytes_silent_audio_returns_empty_or_short(silent_wav_bytes):
    transcript, _ = transcribe_bytes(silent_wav_bytes)
    assert len(transcript) < 50


def test_transcribe_file_returns_tuple(tmp_path):
    wav_path = tmp_path / "test.wav"
    wav_path.write_bytes(_make_silent_wav())
    transcript, confidence = transcribe_file(str(wav_path))
    assert isinstance(transcript, str)
    assert 0.0 <= confidence <= 1.0
