"""API-level schemas — request/response shapes for FastAPI endpoints.

Separated from domain entities to keep API contract independent of internals.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from medai.domain.entities import (
    ApprovalStatus,
    ConfidenceMethod,
    Finding,
    InferenceMetadata,
    JudgmentResult,
    Modality,
    PipelineMetrics,
)


# ═══════════════════════════════════════════════════════════════
#  Case Analysis — Request / Response
# ═══════════════════════════════════════════════════════════════

class CaseAnalysisRequest(BaseModel):
    """Doctor submits a case for AI analysis."""
    patient_id: str
    encounter_id: str | None = None
    image_urls: list[str] = Field(default_factory=list)
    audio_urls: list[str] = Field(default_factory=list)
    clinical_context: str = ""
    doctor_query: str
    patient_history_text: str | None = None
    lab_results: list[dict[str, Any]] | None = None


class CaseAnalysisResponse(BaseModel):
    """AI analysis result returned to the doctor.

    This is the PRIMARY contract between backend and frontend.
    Frontend renders the entire report from this single response.
    """
    report_id: str
    encounter_id: str
    patient_id: str
    diagnosis: str
    confidence: float = Field(description="Overall confidence (see confidence_method for how it was computed)")
    confidence_method: ConfidenceMethod = Field(
        default=ConfidenceMethod.MODEL_SELF_REPORTED,
        description="How the top-level confidence was determined — display as tooltip",
    )
    evidence_summary: str
    timeline_impact: str
    plan: list[str]
    findings: list[Finding]
    reasoning_trace: list[dict[str, Any]]
    judge_verdict: JudgmentResult | None = None
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime
    # Explainability artifacts
    heatmap_urls: list[str] = Field(default_factory=list)
    specialist_summaries: dict[str, str] = Field(default_factory=dict)
    # Performance / provenance
    pipeline_metrics: PipelineMetrics | None = Field(
        default=None,
        description="Timing breakdown for the full pipeline — render in performance panel",
    )


# ═══════════════════════════════════════════════════════════════
#  Report Approval
# ═══════════════════════════════════════════════════════════════

class ReportApprovalRequest(BaseModel):
    """Doctor approves, edits, or rejects an AI report."""
    report_id: str
    status: ApprovalStatus
    doctor_notes: str | None = None
    edits: dict[str, Any] | None = None


class ReportApprovalResponse(BaseModel):
    """Confirmation of doctor's action on a report."""
    report_id: str
    status: ApprovalStatus
    updated_at: datetime


# ═══════════════════════════════════════════════════════════════
#  Patient & Timeline
# ═══════════════════════════════════════════════════════════════

class PatientCreateRequest(BaseModel):
    """Create a new patient record."""
    name: str
    date_of_birth: str  # ISO date
    gender: str = "unknown"
    medical_record_number: str | None = None


class PatientUpdateRequest(BaseModel):
    """Update an existing patient record. All fields optional."""
    name: str | None = None
    date_of_birth: str | None = None  # ISO date
    gender: str | None = None
    medical_record_number: str | None = None


class PatientSummary(BaseModel):
    """Lightweight patient info for lists."""
    id: str
    name: str
    date_of_birth: str
    gender: str
    medical_record_number: str | None = None
    created_at: datetime


class PatientListResponse(BaseModel):
    """List of patient summaries."""
    patients: list[PatientSummary]
    count: int


class TimelineEventResponse(BaseModel):
    """Single timeline entry for the frontend."""
    id: str
    date: datetime
    event_type: str
    summary: str
    source_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PatientTimelineResponse(BaseModel):
    """Full patient timeline for the frontend."""
    patient_id: str
    events: list[TimelineEventResponse]
    count: int


class ReportSummary(BaseModel):
    """Lightweight report info for lists."""
    report_id: str
    encounter_id: str
    diagnosis: str
    confidence: float
    approval_status: str
    created_at: datetime


class PatientReportsResponse(BaseModel):
    """All reports for a patient."""
    patient_id: str
    reports: list[ReportSummary]
    count: int


# ═══════════════════════════════════════════════════════════════
#  Health Check
# ═══════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """API health check response."""
    status: str = "ok"
    version: str
    tools_registered: list[str] = Field(default_factory=list)
    debug: bool = False
    db_connected: bool = True


# ═══════════════════════════════════════════════════════════════
#  Authentication
# ═══════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    """Login credentials."""
    email: str
    password: str


class RegisterRequest(BaseModel):
    """New user registration."""
    email: str
    password: str
    name: str
    role: str = "doctor"


class UserResponse(BaseModel):
    """Public user info (no password hash)."""
    id: str
    email: str
    name: str
    role: str


class AuthResponse(BaseModel):
    """Token + user info returned on login/register."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ═══════════════════════════════════════════════════════════════
#  Transcription (MedASR)
# ═══════════════════════════════════════════════════════════════

class TranscribeRequest(BaseModel):
    """Audio sent from the frontend voice button for speech-to-text."""
    audio_base64: str | None = Field(
        default=None,
        description="Base64-encoded audio bytes (WAV/MP3/FLAC/OGG)",
    )
    audio_url: str | None = Field(
        default=None,
        description="URL to a hosted audio file (alternative to base64)",
    )
    language: str = Field(
        default="en",
        description="Language hint for the ASR model",
    )


class TranscribeResponse(BaseModel):
    """Transcription result returned to the frontend."""
    transcription: str
    duration_seconds: float = 0.0
    warning: str | None = None
    error: str | None = None
