"""Whisper-based transcription route.

POST /api/v1/transcribe
Accepts audio as base64-encoded WAV, transcribes with OpenAI Whisper locally,
and returns the text.  Used by the frontend voice button.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import time

import numpy as np
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = structlog.get_logger()

router = APIRouter(tags=["transcription"])

# ── Lazy-loaded Whisper model (loaded once on first request) ──

_whisper_model = None

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info("loading_whisper_model", model_size=WHISPER_MODEL_SIZE)
        import whisper  # type: ignore[import-untyped]

        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        logger.info("whisper_model_loaded")
    return _whisper_model


# ── Schemas ───────────────────────────────────────────────────

class TranscribeRequest(BaseModel):
    audio_base64: str
    language: str = "en"


class TranscribeResponse(BaseModel):
    transcription: str
    duration_seconds: float
    error: str | None = None


# ── Helpers ───────────────────────────────────────────────────

def _decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes → (float32 ndarray, sample_rate).

    Falls back to raw 16-bit PCM at 16 kHz if WAV header parsing fails.
    """
    from scipy.io import wavfile as scipy_wav  # type: ignore[import-untyped]

    buf = io.BytesIO(audio_bytes)
    try:
        sr, data = scipy_wav.read(buf)
    except Exception:
        # Fallback: treat as raw 16-bit little-endian PCM at 16 kHz
        sr = 16000
        data = np.frombuffer(audio_bytes, dtype=np.int16)

    # → float32
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2_147_483_648.0
    else:
        audio = data.astype(np.float32)

    # → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio, sr


def _resample_if_needed(audio: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample audio to target_sr if necessary."""
    if sr == target_sr:
        return audio
    import math
    from scipy import signal as scipy_signal  # type: ignore[import-untyped]

    gcd = math.gcd(sr, target_sr)
    return scipy_signal.resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)


# ── Route ─────────────────────────────────────────────────────

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(req: TranscribeRequest):
    """Transcribe base64-encoded audio using Whisper."""
    try:
        # Decode base64 → raw bytes
        raw_b64 = req.audio_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]

        # Strip whitespace / newlines that might come from chunked encoding
        raw_b64 = raw_b64.strip()

        try:
            audio_bytes = base64.b64decode(raw_b64)
        except Exception as b64_err:
            logger.error("base64_decode_failed", error=str(b64_err))
            return TranscribeResponse(
                transcription="",
                duration_seconds=0.0,
                error=f"Invalid base64 audio data: {b64_err}",
            )

        if len(audio_bytes) < 44:
            return TranscribeResponse(
                transcription="",
                duration_seconds=0.0,
                error="Audio data too short (less than WAV header size).",
            )

        # Decode WAV → float32 numpy array
        audio, sr = _decode_audio(audio_bytes)

        # Resample to 16 kHz (required by Whisper)
        audio = _resample_if_needed(audio, sr)

        duration = len(audio) / 16000.0
        if duration < 0.1:
            return TranscribeResponse(
                transcription="",
                duration_seconds=round(duration, 2),
                error="Audio is too short to transcribe.",
            )

        logger.info("transcribe_request", duration_s=round(duration, 1), sr=sr)

        # Transcribe with Whisper
        model = _get_model()
        t0 = time.time()

        result = model.transcribe(
            audio,
            language=req.language,
            fp16=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            condition_on_previous_text=True,
        )
        elapsed = time.time() - t0

        text = result.get("text", "").strip()
        logger.info(
            "transcribe_complete",
            text_len=len(text),
            elapsed_ms=round(elapsed * 1000),
        )

        return TranscribeResponse(
            transcription=text,
            duration_seconds=round(duration, 2),
        )

    except Exception as exc:
        logger.error("transcribe_error", error=str(exc), exc_info=True)
        return TranscribeResponse(
            transcription="",
            duration_seconds=0.0,
            error=str(exc),
        )
