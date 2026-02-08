"""FastAPI application entry point.

Assembles routers, middleware, and startup/shutdown hooks.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from medai import __version__
from medai.api.routes import health, cases, patients, auth, transcription
from medai.config import get_settings
from medai.repositories.database import dispose_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle — startup and shutdown hooks."""
    logger = structlog.get_logger()
    settings = get_settings()

    logger.info(
        "starting_medai_backend",
        version=__version__,
        debug=settings.debug,
        model=settings.orchestrator_model,
    )

    # Initialize database tables (dev convenience — use Alembic in prod)
    await init_db()
    logger.info("database_initialized")

    yield  # App is running

    # Clean shutdown
    await dispose_db()
    logger.info("shutting_down_medai_backend")


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI app."""
    settings = get_settings()

    app = FastAPI(
        title="MedAI — Agentic Medical AI Assistant",
        description=(
            "End-to-end medical AI backend with Claude orchestration, "
            "MedGemma specialist tools, and explainable AI reports."
        ),
        version=__version__,
        lifespan=lifespan,
        debug=settings.debug,
    )

    # ── Static Files (local artifacts) ────────────────────
    storage_path = settings.storage_local_path
    storage_path.mkdir(parents=True, exist_ok=True)
    app.mount("/storage", StaticFiles(directory=str(storage_path)), name="storage")

    # ── CORS (restricted to configured origins) ────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ─────────────────────────────────────────────
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(cases.router, prefix="/api/v1")
    app.include_router(patients.router, prefix="/api/v1")
    app.include_router(transcription.router, prefix="/api/v1")

    return app


# Module-level app instance for uvicorn
app = create_app()
