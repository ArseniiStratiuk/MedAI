"""Application configuration via pydantic-settings.

All config is loaded from environment variables / .env file.
Immutable after creation — safe for concurrent access.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class StorageBackend(str, Enum):
    LOCAL = "local"
    S3 = "s3"


class Settings(BaseSettings):
    """Central application settings.

    Reads from environment variables,
    falls back to backend/.env file resolved relative to this module.
    """

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Anthropic ──────────────────────────────────────────
    anthropic_api_key: str = Field(
        ...,
        description="Anthropic API key for Claude orchestrator",
    )
    orchestrator_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Claude model to use for orchestration",
    )
    orchestrator_max_tokens: int = Field(
        default=16384,
        description="Max output tokens per orchestrator call",
    )
    judge_max_tokens: int = Field(
        default=2048,
        description="Max output tokens per judge call",
    )

    # ── Tool Endpoints (Modal / self-hosted) ───────────────
    medgemma_4b_endpoint: str = Field(
        default="http://localhost:8010",
        description="MedGemma 4B IT inference endpoint",
    )
    medgemma_27b_endpoint: str = Field(
        default="http://localhost:8011",
        description="MedGemma 27B Text IT inference endpoint",
    )
    medsiglip_endpoint: str = Field(
        default="http://localhost:8012",
        description="MedSigLIP image encoder endpoint",
    )
    hear_endpoint: str = Field(
        default="http://localhost:8013",
        description="HeAR audio encoder endpoint",
    )
    medasr_endpoint: str = Field(
        default="http://localhost:8014",
        description="MedASR speech-to-text endpoint",
    )

    # ── SigLIP Explainability ──────────────────────────────
    siglip_taxonomy_path: Path = Field(
        default=Path(__file__).parent / "tools" / "condition_taxonomy.json",
        description="Path to per-modality condition label taxonomy JSON",
    )

    # ── Database ───────────────────────────────────────────
    database_url: str = Field(
        default="sqlite+aiosqlite:///./medai.db",
        description="Async database connection URL",
    )

    # ── Authentication ─────────────────────────────────────
    jwt_secret: str = Field(
        default="CHANGE-ME-IN-PRODUCTION",
        description="Secret key for signing JWT tokens",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm",
    )
    access_token_expire_minutes: int = Field(
        default=60,
        description="JWT token expiry in minutes",
    )
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="CORS allowed origins (comma-separated in env)",
    )

    # ── Storage ────────────────────────────────────────────
    storage_backend: StorageBackend = Field(default=StorageBackend.LOCAL)
    storage_local_path: Path = Field(default=Path("./storage"))

    # ── Server ─────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    debug: bool = Field(default=False)

    # ── Budget Guard ───────────────────────────────────────
    max_judgment_cycles: int = Field(
        default=2,
        description="Max re-query cycles per case (budget protection). Set 0 to disable requery.",
    )
    confidence_threshold: float = Field(
        default=0.6,
        description="Below this confidence, findings trigger re-analysis",
    )

    # ── Feature Toggles ───────────────────────────────────
    judge_enabled: bool = Field(
        default=True,
        description="Enable the Judge Agent. When False, pipeline skips judgment and returns consensus.",
    )
    enable_27b_reasoning: bool = Field(
        default=True,
        description="Enable MedGemma 27B text reasoning. When False, text_reasoning tool is not registered.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings accessor — cached after first call."""
    return Settings()  # type: ignore[call-arg]
