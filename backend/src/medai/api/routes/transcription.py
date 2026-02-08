"""Transcription endpoint — speech-to-text via MedASR.

Receives audio (base64 or URL) from the frontend voice button,
forwards it to the deployed MedASR service, and returns plain text
that the frontend can use as a prompt to the agent.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from medai.api.auth import get_current_user
from medai.config import get_settings
from medai.domain.entities import User
from medai.domain.schemas import TranscribeRequest, TranscribeResponse
from medai.services.transcription import TranscriptionService, get_transcription_service

router = APIRouter(prefix="/transcribe", tags=["transcription"])


def _get_transcription_service() -> TranscriptionService:
    """FastAPI dependency — creates the service from settings."""
    return get_transcription_service(get_settings())


@router.post("", response_model=TranscribeResponse)
async def transcribe_audio(
    request: TranscribeRequest,
    _current_user: User = Depends(get_current_user),
    service: TranscriptionService = Depends(_get_transcription_service),
) -> TranscribeResponse:
    """Convert audio to text using MedASR.

    The transcribed text is returned as-is so the frontend can
    inject it as a prompt to the agent chat.
    """
    if not request.audio_base64 and not request.audio_url:
        raise HTTPException(
            status_code=400,
            detail="Provide either audio_base64 or audio_url",
        )

    result = await service.transcribe(
        audio_base64=request.audio_base64,
        audio_url=request.audio_url,
        language=request.language,
    )

    return TranscribeResponse(
        transcription=result.transcription,
        duration_seconds=result.duration_seconds,
        warning=result.warning,
        error=result.error,
    )
