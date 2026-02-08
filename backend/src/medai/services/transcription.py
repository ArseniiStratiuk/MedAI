"""MedASR transcription service — converts audio to text.

Calls the deployed MedASR endpoint (google/medasr) to transcribe
doctor–patient conversations. This is NOT an orchestrator tool;
it's a standalone service used by the /transcribe endpoint so the
frontend voice button can convert speech to a text prompt.

Audio is preprocessed (noise reduction, normalisation, resampling)
before being sent to the model so that recognition quality is as
high as possible even with consumer microphones.
"""

from __future__ import annotations

import base64
import io
import struct
import math
import structlog
import httpx

import numpy as np
from scipy import signal as scipy_signal
from scipy.io import wavfile as scipy_wav

from medai.config import Settings

logger = structlog.get_logger()

# ── Audio preprocessing ───────────────────────────────────

TARGET_SR = 16_000  # MedASR expects 16 kHz


def _decode_audio_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV / raw PCM bytes → (float32 mono waveform, sample_rate)."""
    buf = io.BytesIO(raw)
    try:
        sr, data = scipy_wav.read(buf)
    except Exception:
        # Fallback: assume raw 16-bit PCM at 16 kHz
        sr = TARGET_SR
        data = np.frombuffer(raw, dtype=np.int16)

    # → float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2_147_483_648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    # → mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    return data, sr


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to *target_sr* using polyphase filtering."""
    if orig_sr == target_sr:
        return audio
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return scipy_signal.resample_poly(audio, up, down).astype(np.float32)


def _spectral_noise_gate(
    audio: np.ndarray,
    sr: int,
    *,
    noise_floor_percentile: int = 10,
    n_fft: int = 512,
    reduction_factor: float = 0.8,
) -> np.ndarray:
    """Simple spectral-gating noise reduction.

    1. Compute STFT
    2. Estimate noise floor from the quietest frames
    3. Subtract scaled noise spectrum from each frame
    4. Reconstruct via ISTFT
    """
    hop = n_fft // 4
    _, _, Zxx = scipy_signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Noise floor estimate: mean magnitude of the quietest frames
    frame_energy = magnitude.mean(axis=0)
    threshold = np.percentile(frame_energy, noise_floor_percentile)
    noise_frames = magnitude[:, frame_energy <= threshold]
    if noise_frames.size == 0:
        return audio
    noise_profile = noise_frames.mean(axis=1, keepdims=True)

    # Subtract noise profile (soft gating)
    clean_mag = magnitude - reduction_factor * noise_profile
    clean_mag = np.maximum(clean_mag, 0.0)

    # Reconstruct
    clean_Zxx = clean_mag * np.exp(1j * phase)
    _, clean_audio = scipy_signal.istft(clean_Zxx, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    return clean_audio[: len(audio)].astype(np.float32)


def _normalize_volume(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Peak-normalize to *target_db* dBFS."""
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio
    target_peak = 10 ** (target_db / 20.0)
    return (audio * (target_peak / peak)).astype(np.float32)


def _trim_silence(
    audio: np.ndarray,
    sr: int,
    *,
    threshold_db: float = -40.0,
    frame_ms: int = 20,
    pad_ms: int = 150,
) -> np.ndarray:
    """Trim leading/trailing silence based on frame energy."""
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return audio

    threshold_amp = 10 ** (threshold_db / 20.0)

    frame_energies = np.array([
        np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])

    active = np.where(frame_energies > threshold_amp)[0]
    if len(active) == 0:
        return audio

    pad_frames = max(1, int(pad_ms / frame_ms))
    start_frame = max(0, active[0] - pad_frames)
    end_frame = min(n_frames, active[-1] + pad_frames + 1)

    return audio[start_frame * frame_len : end_frame * frame_len]


def _highpass_filter(audio: np.ndarray, sr: int, cutoff: int = 80) -> np.ndarray:
    """Remove low-frequency rumble below *cutoff* Hz."""
    sos = scipy_signal.butter(5, cutoff, btype="high", fs=sr, output="sos")
    return scipy_signal.sosfilt(sos, audio).astype(np.float32)


def preprocess_audio(audio_bytes: bytes) -> str:
    """Full preprocessing pipeline: decode → clean → re-encode as base64 WAV.

    Steps:
    1. Decode incoming audio (WAV or raw PCM)
    2. Resample to 16 kHz
    3. High-pass filter (remove rumble < 80 Hz)
    4. Spectral noise gate (reduce background noise)
    5. Trim silence
    6. Peak-normalize volume
    7. Re-encode as 16-bit WAV base64
    """
    audio, sr = _decode_audio_bytes(audio_bytes)
    logger.info("preprocess_audio", orig_sr=sr, orig_len=len(audio))

    audio = _resample(audio, sr, TARGET_SR)
    audio = _highpass_filter(audio, TARGET_SR)
    audio = _spectral_noise_gate(audio, TARGET_SR)
    audio = _trim_silence(audio, TARGET_SR)
    audio = _normalize_volume(audio)

    # Re-encode as 16-bit PCM WAV → base64
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    scipy_wav.write(buf, TARGET_SR, audio_int16)
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode("ascii")

    logger.info(
        "preprocess_audio_done",
        duration_s=round(len(audio) / TARGET_SR, 2),
        base64_len=len(result),
    )
    return result


class TranscriptionService:
    """Calls the remote MedASR endpoint to transcribe audio."""

    def __init__(self, endpoint: str, timeout: float = 120.0, max_retries: int = 2):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    async def transcribe(
        self,
        *,
        audio_base64: str | None = None,
        audio_url: str | None = None,
        language: str = "en",
    ) -> TranscriptionResult:
        """Send audio to MedASR and return the transcription.

        Provide exactly one of ``audio_base64`` or ``audio_url``.

        When *audio_base64* is provided the audio is decoded,
        preprocessed (resampled, denoised, normalized), and
        re-encoded before being forwarded to MedASR.

        Returns a ``TranscriptionResult`` with the transcribed text,
        duration, and any warnings/errors from the model.
        """
        if not audio_base64 and not audio_url:
            raise ValueError("Provide audio_base64 or audio_url")

        # ── Preprocess base64 audio for better recognition ──
        if audio_base64:
            try:
                # Strip data-URL prefix if present
                raw_b64 = audio_base64
                if "," in raw_b64:
                    raw_b64 = raw_b64.split(",", 1)[1]
                raw_bytes = base64.b64decode(raw_b64)
                audio_base64 = preprocess_audio(raw_bytes)
                logger.info("audio_preprocessed")
            except Exception as exc:
                logger.warning("preprocess_failed_using_raw", error=str(exc))
                # Fall through with original audio_base64

        payload: dict[str, str] = {"language": language}
        if audio_base64:
            payload["audio_base64"] = audio_base64
        else:
            payload["audio_url"] = audio_url  # type: ignore[assignment]

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):  # +2 because range is [1, max+2)
            try:
                logger.info(
                    "medasr_request",
                    attempt=attempt,
                    has_base64=bool(audio_base64),
                    has_url=bool(audio_url),
                )
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(self.endpoint, json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                if "error" in data and data["error"]:
                    logger.warning("medasr_model_error", error=data["error"])
                    return TranscriptionResult(
                        transcription="",
                        duration_seconds=data.get("duration_seconds", 0.0),
                        error=data["error"],
                    )

                return TranscriptionResult(
                    transcription=data.get("transcription", ""),
                    duration_seconds=data.get("duration_seconds", 0.0),
                    inference_time_ms=data.get("inference_time_ms"),
                    model_id=data.get("model_id"),
                    warning=data.get("warning"),
                )

            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_error = exc
                logger.warning(
                    "medasr_retry",
                    attempt=attempt,
                    error=str(exc),
                )

        logger.error("medasr_failed", error=str(last_error))
        return TranscriptionResult(
            transcription="",
            duration_seconds=0.0,
            error=f"MedASR unavailable after {self.max_retries + 1} attempts: {last_error}",
        )


class TranscriptionResult:
    """Simple value object for transcription results."""

    __slots__ = (
        "transcription",
        "duration_seconds",
        "inference_time_ms",
        "model_id",
        "warning",
        "error",
    )

    def __init__(
        self,
        transcription: str,
        duration_seconds: float,
        inference_time_ms: float | None = None,
        model_id: str | None = None,
        warning: str | None = None,
        error: str | None = None,
    ):
        self.transcription = transcription
        self.duration_seconds = duration_seconds
        self.inference_time_ms = inference_time_ms
        self.model_id = model_id
        self.warning = warning
        self.error = error


def get_transcription_service(settings: Settings) -> TranscriptionService:
    """Factory — create a TranscriptionService from app settings."""
    return TranscriptionService(endpoint=settings.medasr_endpoint)
