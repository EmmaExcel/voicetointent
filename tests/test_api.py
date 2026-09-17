from __future__ import annotations

import io
import wave
from unittest.mock import patch, MagicMock

import numpy as np
from fastapi.testclient import TestClient

from main import app
from schemas.financial import TransactionIntent


client = TestClient(app)


MOCK_INTENT = TransactionIntent(
    intent="send_money",
    amount=5000,
    currency="NGN",
    recipient={
        "id": "ben_001",
        "full_name": "Jane Doe",
        "account_number": "0123456789",
        "bank_name": "Guaranty Trust Bank (GTB)",
    },
)


def _make_wav_bytes(duration_seconds: float = 0.3) -> bytes:
    sample_rate = 16000
    num_samples = int(sample_rate * duration_seconds)
    audio = np.zeros(num_samples, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_get_schemas():
    response = client.get("/schemas")
    assert response.status_code == 200
    assert "financial" in response.json()["schemas"]


def test_extract_valid():
    mock_provider = MagicMock()
    mock_provider.extract.return_value = MOCK_INTENT

    with patch("main.get_llm_provider", return_value=mock_provider):
        response = client.post(
            "/extract",
            json={"text": "Send 5k to mum", "schema_name": "financial"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "send_money"
    assert data["amount"] == 5000


def test_extract_unknown_schema():
    response = client.post(
        "/extract",
        json={"text": "test", "schema_name": "nonexistent"},
    )
    assert response.status_code == 404


def test_transcribe_valid_audio():
    wav_bytes = _make_wav_bytes()

    with patch("main.transcriber.transcribe_file", return_value=("send 5k to mum", 0.92)):
        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "transcript" in data
    assert "confidence" in data
    assert data["confidence"] == 0.92


def test_transcribe_rejects_oversized_file():
    large_bytes = b"x" * (26 * 1024 * 1024)
    response = client.post(
        "/transcribe",
        files={"audio": ("big.wav", large_bytes, "audio/wav")},
    )
    assert response.status_code == 413


def test_process_full_pipeline():
    wav_bytes = _make_wav_bytes()
    mock_provider = MagicMock()
    mock_provider.extract.return_value = MOCK_INTENT

    with (
        patch("main.transcriber.transcribe_file", return_value=("send 5k to mum", 0.88)),
        patch("main.get_llm_provider", return_value=mock_provider),
    ):
        response = client.post(
            "/process",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"schema_name": "financial"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["transcript"] == "send 5k to mum"
    assert data["confidence"] == 0.88
    assert data["intent"]["intent"] == "send_money"


def test_process_unknown_schema():
    wav_bytes = _make_wav_bytes()
    response = client.post(
        "/process",
        files={"audio": ("test.wav", wav_bytes, "audio/wav")},
        data={"schema_name": "unknown"},
    )
    assert response.status_code == 404
