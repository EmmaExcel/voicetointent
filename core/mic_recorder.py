from __future__ import annotations

import io
import queue
import time
import wave
from threading import Event

import numpy as np
import sounddevice as sd

from config import MicConfig


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) / 32768.0


def _rms_bar(rms: float, width: int = 20) -> str:
    filled = int(min(rms / 0.1, 1.0) * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {rms:.4f}"


def record_until_silence(
    silence_seconds: float | None = None,
    silence_rms: float | None = None,
    sample_rate: int | None = None,
    max_duration: float = 30.0,
    stop_event: Event | None = None,
    show_levels: bool = True,
) -> bytes:
    silence_seconds = silence_seconds or MicConfig.silence_threshold_seconds
    silence_rms = silence_rms or MicConfig.silence_rms_threshold
    sample_rate = sample_rate or MicConfig.sample_rate

    block_size = int(sample_rate * 0.05)
    silence_blocks_needed = int(silence_seconds / 0.05)

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = stop_event or Event()

    def _callback(indata: np.ndarray, frames: int, t, status):
        if status:
            print(f"[mic] warning: {status}")
        audio_queue.put(indata.copy())

    all_frames: list[np.ndarray] = []
    silence_blocks = 0
    speech_detected = False
    start_time = time.time()

    print("[mic] listening... (speak now, Ctrl+C to cancel)")

    if show_levels:
        print("[mic] live levels (0.1+ = speech):")

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=block_size,
            callback=_callback,
        ):
            while not stop_event.is_set():
                elapsed = time.time() - start_time

                if elapsed > max_duration:
                    print(f"\n[mic] max duration ({max_duration}s) reached, stopping.")
                    break

                try:
                    frame = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                all_frames.append(frame)
                amplitude = _rms(frame)

                if show_levels:
                    print(f"\r[mic] {_rms_bar(amplitude)}  ", end="", flush=True)

                if amplitude > silence_rms:
                    speech_detected = True
                    silence_blocks = 0
                elif speech_detected:
                    silence_blocks += 1
                    if silence_blocks >= silence_blocks_needed:
                        print("\n[mic] silence detected, stopping.")
                        break

    except KeyboardInterrupt:
        print("\n[mic] recording cancelled.")
        return b""

    if not all_frames:
        print("[mic] no audio captured. check microphone permissions.")
        return b""

    if not speech_detected:
        print(
            "[mic] no speech detected.\n"
            "  - check System Settings > Privacy & Security > Microphone\n"
            "  - try lowering MIC_SILENCE_RMS_THRESHOLD in .env (e.g. 0.005)"
        )

    audio_data = np.concatenate(all_frames, axis=0)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    return wav_buffer.getvalue()
