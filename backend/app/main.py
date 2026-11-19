"""
app/main.py

Responsibility: FastAPI application factory. Registers routers, CORS middleware,
exception handlers, and startup/shutdown events. Import this module to get the
`app` object for uvicorn.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.db import init_db

# ---------------------------------------------------------------------------
# Logging must be configured before anything else logs
# ---------------------------------------------------------------------------
_settings = get_settings()
setup_logging(level=_settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    init_db()
    logger.info(
        "App started",
        extra={
            "provider": settings.app.provider,
            "version": settings.app.version,
        },
    )
    yield
    logger.info("App shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Support Triage Agent API",
        description=(
            "Multi-agent support desk backend: Triage → Resolver → Critic graph "
            "with human-in-the-loop approval and eval gate."
        ),
        version=settings.app.version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS — allow the Vite dev server and production frontend origins ────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",   # Vite default dev port
            "http://localhost:4173",   # Vite preview port
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ───────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ─────────────────────────────────────────────────────────────
    from app.api.routes import health, kb, tickets  # noqa: PLC0415

    app.include_router(health.router, prefix="/api")
    app.include_router(kb.router, prefix="/api")
    app.include_router(tickets.router, prefix="/api")

    # Placeholder routers — added in later phases
    # app.include_router(tickets.router, prefix="/api")
    # app.include_router(approvals.router, prefix="/api")
    # app.include_router(kb.router, prefix="/api")
    # app.include_router(policies.router, prefix="/api")
    # app.include_router(evals.router, prefix="/api")
    # app.include_router(traces.router, prefix="/api")

    return app


app = create_app()
