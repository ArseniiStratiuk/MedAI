"""Unit tests for the transcription service and /transcribe endpoint.

Tests cover:
- TranscriptionService: success, model-level error, HTTP failures, retries
- /api/v1/transcribe endpoint: validation, auth, integration with service
- Audio preprocessing pipeline: decode, resample, filter, noise gate, normalize
"""

from __future__ import annotations

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key-not-real")
os.environ.setdefault("DEBUG", "true")

import base64
import io
import json
from datetime import date, datetime
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import pytest
import pytest_asyncio
import respx
from httpx import ASGITransport, AsyncClient
from scipy.io import wavfile as scipy_wav
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from medai.api.auth import get_current_user
from medai.domain.entities import User, UserRole
from medai.domain.schemas import TranscribeRequest, TranscribeResponse
from medai.main import create_app
from medai.repositories.database import get_db_session
from medai.repositories.models import Base
from medai.services.transcription import (
    TARGET_SR,
    TranscriptionResult,
    TranscriptionService,
    _decode_audio_bytes,
    _highpass_filter,
    _normalize_volume,
    _resample,
    _spectral_noise_gate,
    _trim_silence,
    preprocess_audio,
)

MOCK_MEDASR = "http://mock-medasr:9000"

_TEST_USER = User(
    id="USR-TEST0001",
    email="test@medai.com",
    hashed_password="not-used",
    name="Test Doctor",
    role=UserRole.DOCTOR,
)


# ── helpers ───────────────────────────────────────────────────

def _make_wav_base64(sr: int = 16000, duration: float = 1.0, freq: float = 440.0) -> str:
    """Create a base64-encoded WAV containing a sine tone."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    audio_int16 = (tone * 32767).astype(np.int16)
    buf = io.BytesIO()
    scipy_wav.write(buf, sr, audio_int16)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _make_wav_bytes(sr: int = 16000, duration: float = 0.5, freq: float = 440.0) -> bytes:
    """Create raw WAV bytes containing a sine tone."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    audio_int16 = (tone * 32767).astype(np.int16)
    buf = io.BytesIO()
    scipy_wav.write(buf, sr, audio_int16)
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def service() -> TranscriptionService:
    return TranscriptionService(endpoint=MOCK_MEDASR, max_retries=1)


# Patch preprocessing in service tests so we can use trivial base64 payloads
# without needing valid WAV data.  The preprocessing pipeline is tested
# separately in TestPreprocessing below.
_PATCH_PREPROCESS = patch(
    "medai.services.transcription.preprocess_audio",
    side_effect=lambda raw: base64.b64encode(raw).decode(),
)


@pytest_asyncio.fixture
async def _db_session_factory():
    def _json_serializer(obj):
        def default(o):
            if isinstance(o, (datetime, date)):
                return o.isoformat()
            raise TypeError(
                f"Object of type {type(o).__name__} is not JSON serializable"
            )

        return json.dumps(obj, default=default)

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        json_serializer=_json_serializer,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    yield factory
    await engine.dispose()


@pytest.fixture
def app(_db_session_factory):
    application = create_app()

    async def _override_db_session():
        async with _db_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    application.dependency_overrides[get_db_session] = _override_db_session

    async def _override_current_user() -> User:
        return _TEST_USER

    application.dependency_overrides[get_current_user] = _override_current_user

    return application


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ═══════════════════════════════════════════════════════════════
#  TranscriptionService — unit tests
# ═══════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestTranscriptionService:
    """Tests for the service layer that calls the MedASR endpoint."""

    @pytest.mark.asyncio
    @respx.mock
    @_PATCH_PREPROCESS
    async def test_successful_transcription_with_base64(self, _mock_pp, service):
        respx.post(MOCK_MEDASR).mock(
            return_value=httpx.Response(
                200,
                json={
                    "transcription": "Patient reports chest pain for two days.",
                    "duration_seconds": 5.3,
                    "inference_time_ms": 420.5,
                    "model_id": "google/medasr",
                },
            )
        )

        result = await service.transcribe(audio_base64="dGVzdA==")

        assert result.transcription == "Patient reports chest pain for two days."
        assert result.duration_seconds == 5.3
        assert result.inference_time_ms == 420.5
        assert result.model_id == "google/medasr"
        assert result.error is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_transcription_with_url(self, service):
        respx.post(MOCK_MEDASR).mock(
            return_value=httpx.Response(
                200,
                json={
                    "transcription": "No acute distress noted.",
                    "duration_seconds": 2.1,
                    "model_id": "google/medasr",
                },
            )
        )

        result = await service.transcribe(
            audio_url="https://example.com/audio.wav"
        )

        assert result.transcription == "No acute distress noted."
        assert result.duration_seconds == 2.1
        assert result.error is None

    @pytest.mark.asyncio
    @respx.mock
    @_PATCH_PREPROCESS
    async def test_model_returns_error(self, _mock_pp, service):
        """MedASR returns 200 but with an error field (e.g. audio too short)."""
        respx.post(MOCK_MEDASR).mock(
            return_value=httpx.Response(
                200,
                json={
                    "transcription": "",
                    "duration_seconds": 0.05,
                    "error": "Audio too short after silence removal",
                },
            )
        )

        result = await service.transcribe(audio_base64="dGVzdA==")

        assert result.transcription == ""
        assert result.error == "Audio too short after silence removal"

    @pytest.mark.asyncio
    @respx.mock
    @_PATCH_PREPROCESS
    async def test_model_returns_warning(self, _mock_pp, service):
        """MedASR returns a result with a warning."""
        respx.post(MOCK_MEDASR).mock(
            return_value=httpx.Response(
                200,
                json={
                    "transcription": "uh",
                    "duration_seconds": 0.3,
                    "model_id": "google/medasr",
                    "warning": "Audio too short after silence removal",
                },
            )
        )

        result = await service.transcribe(audio_base64="dGVzdA==")

        assert result.transcription == "uh"
        assert result.warning == "Audio too short after silence removal"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_missing_audio_raises(self, service):
        with pytest.raises(ValueError, match="audio_base64 or audio_url"):
            await service.transcribe()

    @pytest.mark.asyncio
    @respx.mock
    @_PATCH_PREPROCESS
    async def test_http_error_retries_and_fails(self, _mock_pp, service):
        """Service should retry on HTTP errors and return an error result."""
        respx.post(MOCK_MEDASR).mock(
            return_value=httpx.Response(503, text="Service Unavailable")
        )

        result = await service.transcribe(audio_base64="dGVzdA==")

        assert result.transcription == ""
        assert result.error is not None
        assert "unavailable" in result.error.lower() or "503" in result.error

    @pytest.mark.asyncio
    @respx.mock
    @_PATCH_PREPROCESS
    async def test_http_timeout_retries(self, _mock_pp, service):
        """Service should retry on connection errors."""
        respx.post(MOCK_MEDASR).mock(side_effect=httpx.ConnectTimeout("timeout"))

        result = await service.transcribe(audio_base64="dGVzdA==")

        assert result.transcription == ""
        assert result.error is not None

    @pytest.mark.asyncio
    @respx.mock
    @_PATCH_PREPROCESS
    async def test_retry_succeeds_on_second_attempt(self, _mock_pp, service):
        """First attempt fails, second succeeds."""
        route = respx.post(MOCK_MEDASR)
        route.side_effect = [
            httpx.Response(503, text="Service Unavailable"),
            httpx.Response(
                200,
                json={
                    "transcription": "Recovery successful.",
                    "duration_seconds": 1.0,
                    "model_id": "google/medasr",
                },
            ),
        ]

        result = await service.transcribe(audio_base64="dGVzdA==")

        assert result.transcription == "Recovery successful."
        assert result.error is None

    @pytest.mark.asyncio
    @respx.mock
    @_PATCH_PREPROCESS
    async def test_empty_transcription_is_valid(self, _mock_pp, service):
        """Empty transcription (silence) is a valid response, not an error."""
        respx.post(MOCK_MEDASR).mock(
            return_value=httpx.Response(
                200,
                json={
                    "transcription": "",
                    "duration_seconds": 3.0,
                    "model_id": "google/medasr",
                },
            )
        )

        result = await service.transcribe(audio_base64="dGVzdA==")

        assert result.transcription == ""
        assert result.error is None


# ═══════════════════════════════════════════════════════════════
#  /api/v1/transcribe endpoint — integration tests
# ═══════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestTranscribeEndpoint:
    """Tests for the FastAPI /transcribe route."""

    @pytest.mark.asyncio
    async def test_transcribe_with_base64(self, client):
        """Happy path — audio_base64 provided, MedASR returns transcription."""
        mock_result = TranscriptionResult(
            transcription="Patient has a persistent cough.",
            duration_seconds=4.2,
            inference_time_ms=310.0,
            model_id="google/medasr",
        )

        from medai.api.routes.transcription import _get_transcription_service

        svc = AsyncMock(spec=TranscriptionService)
        svc.transcribe.return_value = mock_result
        client._transport.app.dependency_overrides[  # type: ignore[attr-defined]
            _get_transcription_service
        ] = lambda: svc

        resp = await client.post(
            "/api/v1/transcribe",
            json={"audio_base64": "dGVzdA=="},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["transcription"] == "Patient has a persistent cough."
        assert data["duration_seconds"] == 4.2
        assert data["error"] is None

    @pytest.mark.asyncio
    async def test_transcribe_with_url(self, client):
        """Happy path — audio_url provided."""
        mock_result = TranscriptionResult(
            transcription="Breath sounds diminished bilaterally.",
            duration_seconds=6.0,
        )

        from medai.api.routes.transcription import _get_transcription_service

        svc = AsyncMock(spec=TranscriptionService)
        svc.transcribe.return_value = mock_result
        client._transport.app.dependency_overrides[  # type: ignore[attr-defined]
            _get_transcription_service
        ] = lambda: svc

        resp = await client.post(
            "/api/v1/transcribe",
            json={"audio_url": "https://example.com/audio.wav"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["transcription"] == "Breath sounds diminished bilaterally."

    @pytest.mark.asyncio
    async def test_transcribe_no_audio_returns_400(self, client):
        """Missing both audio_base64 and audio_url should 400."""
        resp = await client.post(
            "/api/v1/transcribe",
            json={},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_transcribe_returns_error_from_service(self, client):
        """If MedASR itself reports an error, relay it to the frontend."""
        mock_result = TranscriptionResult(
            transcription="",
            duration_seconds=0.05,
            error="Audio too short after silence removal",
        )

        from medai.api.routes.transcription import _get_transcription_service

        svc = AsyncMock(spec=TranscriptionService)
        svc.transcribe.return_value = mock_result
        client._transport.app.dependency_overrides[  # type: ignore[attr-defined]
            _get_transcription_service
        ] = lambda: svc

        resp = await client.post(
            "/api/v1/transcribe",
            json={"audio_base64": "dGVzdA=="},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["transcription"] == ""
        assert data["error"] == "Audio too short after silence removal"

    @pytest.mark.asyncio
    async def test_transcribe_with_language_param(self, client):
        """Language hint is forwarded to the service."""
        mock_result = TranscriptionResult(
            transcription="El paciente reporta dolor.",
            duration_seconds=3.0,
        )

        from medai.api.routes.transcription import _get_transcription_service

        svc = AsyncMock(spec=TranscriptionService)
        svc.transcribe.return_value = mock_result
        client._transport.app.dependency_overrides[  # type: ignore[attr-defined]
            _get_transcription_service
        ] = lambda: svc

        resp = await client.post(
            "/api/v1/transcribe",
            json={"audio_base64": "dGVzdA==", "language": "es"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["transcription"] == "El paciente reporta dolor."
        svc.transcribe.assert_called_once_with(
            audio_base64="dGVzdA==",
            audio_url=None,
            language="es",
        )


# ═══════════════════════════════════════════════════════════════
#  Audio preprocessing — unit tests
# ═══════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPreprocessing:
    """Tests for the audio preprocessing pipeline used before sending to MedASR."""

    def test_decode_wav_int16(self):
        """Decode a 16-bit WAV and get float32 mono."""
        wav_bytes = _make_wav_bytes(sr=16000, duration=0.1)
        audio, sr = _decode_audio_bytes(wav_bytes)
        assert sr == 16000
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert len(audio) > 0

    def test_decode_wav_preserves_sample_rate(self):
        """Verify the detected sample rate matches what was encoded."""
        wav_bytes = _make_wav_bytes(sr=48000, duration=0.1)
        audio, sr = _decode_audio_bytes(wav_bytes)
        assert sr == 48000

    def test_decode_raw_pcm_fallback(self):
        """Non-WAV bytes → raw 16-bit PCM at 16 kHz assumed."""
        # 100 samples of int16 zeros
        raw = np.zeros(100, dtype=np.int16).tobytes()
        audio, sr = _decode_audio_bytes(raw)
        assert sr == TARGET_SR
        assert len(audio) == 100

    def test_resample_noop_same_rate(self):
        """No-op when source and target rates match."""
        audio = np.random.randn(16000).astype(np.float32)
        out = _resample(audio, 16000, 16000)
        np.testing.assert_array_equal(audio, out)

    def test_resample_48k_to_16k(self):
        """48 kHz → 16 kHz should produce 1/3 as many samples."""
        audio = np.random.randn(48000).astype(np.float32)
        out = _resample(audio, 48000, 16000)
        assert abs(len(out) - 16000) < 10  # allow small rounding

    def test_highpass_removes_dc(self):
        """High-pass filter should remove DC offset."""
        audio = np.ones(16000, dtype=np.float32) * 0.5  # DC signal
        filtered = _highpass_filter(audio, 16000, cutoff=80)
        # DC should be mostly removed
        assert abs(float(np.mean(filtered))) < 0.05

    def test_normalize_volume_quiet_signal(self):
        """Normalize a quiet signal to target peak."""
        audio = np.ones(1000, dtype=np.float32) * 0.01
        normed = _normalize_volume(audio, target_db=-3.0)
        peak = float(np.max(np.abs(normed)))
        expected_peak = 10 ** (-3.0 / 20.0)
        assert abs(peak - expected_peak) < 0.01

    def test_normalize_volume_silent(self):
        """Silent audio should be returned unchanged to avoid divide-by-zero."""
        audio = np.zeros(1000, dtype=np.float32)
        normed = _normalize_volume(audio)
        np.testing.assert_array_equal(audio, normed)

    def test_trim_silence_removes_padding(self):
        """Leading/trailing silence should be trimmed."""
        sr = 16000
        silence = np.zeros(sr, dtype=np.float32)  # 1s silence
        tone = (np.sin(np.linspace(0, 440 * 2 * np.pi, sr)) * 0.5).astype(np.float32)
        audio = np.concatenate([silence, tone, silence])  # 3s total
        trimmed = _trim_silence(audio, sr)
        # Should be shorter than original
        assert len(trimmed) < len(audio)
        # Should still contain most of the tone
        assert len(trimmed) >= sr * 0.8

    def test_spectral_noise_gate_preserves_signal(self):
        """Noise gating should not destroy a signal with some noise."""
        sr = 16000
        t = np.linspace(0, 1, sr, endpoint=False)
        # Loud speech-like tone with quiet noise floor at start/end
        noise = np.random.randn(sr).astype(np.float32) * 0.01
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        # 0.2s noise, 0.6s tone, 0.2s noise
        audio = np.concatenate([noise[:3200], tone[3200:12800], noise[12800:]])
        cleaned = _spectral_noise_gate(audio, sr)
        # The tone portion should retain significant energy
        tone_slice_orig = float(np.sum(audio[3200:12800] ** 2))
        tone_slice_clean = float(np.sum(cleaned[3200:12800] ** 2))
        assert tone_slice_clean > tone_slice_orig * 0.3

    def test_preprocess_audio_full_pipeline(self):
        """Full pipeline: WAV bytes → base64 WAV at 16 kHz."""
        wav_bytes = _make_wav_bytes(sr=48000, duration=0.5, freq=440.0)
        result_b64 = preprocess_audio(wav_bytes)

        # Should be valid base64
        decoded = base64.b64decode(result_b64)
        assert len(decoded) > 44  # at least a WAV header

        # Should be a valid WAV at 16 kHz
        buf = io.BytesIO(decoded)
        sr, data = scipy_wav.read(buf)
        assert sr == TARGET_SR
        assert data.dtype == np.int16
        assert len(data) > 0

    def test_preprocess_audio_handles_16k_input(self):
        """16 kHz input should pass through without resampling errors."""
        wav_bytes = _make_wav_bytes(sr=16000, duration=0.3)
        result_b64 = preprocess_audio(wav_bytes)
        decoded = base64.b64decode(result_b64)
        buf = io.BytesIO(decoded)
        sr, data = scipy_wav.read(buf)
        assert sr == TARGET_SR

    @pytest.mark.asyncio
    @respx.mock
    async def test_service_preprocesses_base64_before_sending(self):
        """Verify the service calls preprocessing on audio_base64."""
        wav_b64 = _make_wav_base64(sr=48000, duration=0.3)

        respx.post(MOCK_MEDASR).mock(
            return_value=httpx.Response(
                200,
                json={
                    "transcription": "Preprocessed OK.",
                    "duration_seconds": 0.3,
                    "model_id": "google/medasr",
                },
            )
        )

        svc = TranscriptionService(endpoint=MOCK_MEDASR, max_retries=0)
        result = await svc.transcribe(audio_base64=wav_b64)

        assert result.transcription == "Preprocessed OK."
        assert result.error is None

        # Verify the payload sent to MedASR has preprocessed (16kHz) audio
        req = respx.calls.last.request
        import json as _json

        body = _json.loads(req.content)
        sent_b64 = body["audio_base64"]
        raw = base64.b64decode(sent_b64)
        buf = io.BytesIO(raw)
        sr, _ = scipy_wav.read(buf)
        assert sr == TARGET_SR  # was 48k, now 16k

    @pytest.mark.asyncio
    @respx.mock
    async def test_service_skips_preprocess_for_url(self):
        """audio_url should not be preprocessed — it's fetched by MedASR."""
        respx.post(MOCK_MEDASR).mock(
            return_value=httpx.Response(
                200,
                json={
                    "transcription": "URL audio.",
                    "duration_seconds": 1.0,
                    "model_id": "google/medasr",
                },
            )
        )

        svc = TranscriptionService(endpoint=MOCK_MEDASR, max_retries=0)
        result = await svc.transcribe(audio_url="https://example.com/audio.wav")

        assert result.transcription == "URL audio."
        req = respx.calls.last.request
        import json as _json

        body = _json.loads(req.content)
        assert "audio_url" in body
        assert body["audio_url"] == "https://example.com/audio.wav"
