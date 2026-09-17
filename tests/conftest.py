from __future__ import annotations

import io
import wave

import numpy as np
import pytest


@pytest.fixture
def silent_wav_bytes() -> bytes:
    sample_rate = 16000
    duration_seconds = 0.5
    num_samples = int(sample_rate * duration_seconds)
    audio = np.zeros(num_samples, dtype=np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return buf.getvalue()


@pytest.fixture
def sample_transcript() -> str:
    return "Send 5k to mum"


@pytest.fixture
def mock_intent() -> dict:
    return {
        "intent": "send_money",
        "amount": 5000,
        "currency": "NGN",
        "recipient": {
            "id": "ben_001",
            "full_name": "Jane Doe",
            "account_number": "0123456789",
            "bank_name": "Guaranty Trust Bank (GTB)",
        },
    }
