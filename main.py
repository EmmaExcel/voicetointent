from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_llm_provider
from core import transcriber
from schemas import get_schema, list_schemas


ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/ogg",
    "audio/flac",
    "audio/webm",
    "application/octet-stream",
}

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB


app = FastAPI(
    title="Voice-to-Intent API",
    description=(
        "Transcribe audio and extract structured financial intent "
        "using a configurable LLM backend."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptResponse(BaseModel):
    transcript: str
    confidence: float


class ExtractRequest(BaseModel):
    text: str
    schema_name: str = "financial"
    system_prompt: str = ""


class ProcessResponse(BaseModel):
    transcript: str
    confidence: float
    schema_name: str
    intent: dict


def _validate_audio(audio: UploadFile) -> None:
    content_type = audio.content_type or ""
    if content_type and content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{content_type}'. Expected an audio file.",
        )


async def _read_audio(audio: UploadFile) -> bytes:
    data = await audio.read()
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file exceeds the 25MB limit ({len(data)} bytes received).",
        )
    return data


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "voice-to-intent"}


@app.get("/schemas", tags=["Schemas"])
def get_schemas():
    return {"schemas": list_schemas()}


@app.post("/transcribe", response_model=TranscriptResponse, tags=["Pipeline"])
async def transcribe_audio(audio: UploadFile = File(...)):
    _validate_audio(audio)
    data = await _read_audio(audio)

    suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        text, confidence = transcriber.transcribe_file(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        os.unlink(tmp_path)

    return TranscriptResponse(transcript=text, confidence=confidence)


@app.post("/extract", tags=["Pipeline"])
def extract_intent(req: ExtractRequest):
    try:
        schema_cls = get_schema(req.schema_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        provider = get_llm_provider()
        result = provider.extract(req.text, schema_cls, req.system_prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")

    return result.model_dump()


@app.post("/process", response_model=ProcessResponse, tags=["Pipeline"])
async def process_audio(
    audio: UploadFile = File(...),
    schema_name: str = Form("financial"),
    system_prompt: str = Form(""),
):
    _validate_audio(audio)

    try:
        schema_cls = get_schema(schema_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    data = await _read_audio(audio)
    suffix = Path(audio.filename).suffix if audio.filename else ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        transcript, confidence = transcriber.transcribe_file(tmp_path)

        if not transcript:
            raise HTTPException(
                status_code=422,
                detail="No speech detected in the audio.",
            )

        provider = get_llm_provider()
        intent_obj = provider.extract(transcript, schema_cls, system_prompt)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    return ProcessResponse(
        transcript=transcript,
        confidence=confidence,
        schema_name=schema_name,
        intent=intent_obj.model_dump(),
    )


@app.websocket("/ws/process")
async def ws_process(websocket: WebSocket, schema_name: str = "financial"):
    await websocket.accept()

    try:
        schema_cls = get_schema(schema_name)
    except KeyError as exc:
        await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        await websocket.close()
        return

    try:
        audio_chunks: list[bytes] = []

        while True:
            try:
                chunk = await websocket.receive_bytes()
                audio_chunks.append(chunk)
            except WebSocketDisconnect:
                break

        if not audio_chunks:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "No audio received."})
            )
            return

        audio_bytes = b"".join(audio_chunks)
        transcript, confidence = transcriber.transcribe_bytes(audio_bytes)

        if not transcript:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "No speech detected."})
            )
            return

        await websocket.send_text(
            json.dumps({"type": "transcript", "text": transcript, "confidence": confidence})
        )

        provider = get_llm_provider()
        intent_obj = provider.extract(transcript, schema_cls)

        await websocket.send_text(
            json.dumps({
                "type": "intent",
                "data": intent_obj.model_dump(),
                "confidence": confidence,
            })
        )

    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
    finally:
        await websocket.close()
