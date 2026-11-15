"""
app/api/routes/health.py

Responsibility: /api/health endpoint — returns current provider status,
active models, and app version. Used by the frontend Health page and
uptime monitors.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Shape of the /api/health response."""

    status: str         # "ok" | "degraded" | "error"
    provider: str
    chat_model: str
    embed_model: str
    version: str


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    """
    Return the current health status and active provider configuration.
    Does NOT make a live call to the LLM — just validates config is loadable.
    """
    try:
        settings = get_settings()
        app_cfg = settings.app
        provider_cfg = settings.provider
        return HealthResponse(
            status="ok",
            provider=app_cfg.provider,
            chat_model=provider_cfg.llm.model,
            embed_model=provider_cfg.embedder.model,
            version=app_cfg.version,
        )
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return HealthResponse(
            status="error",
            provider="unknown",
            chat_model="unknown",
            embed_model="unknown",
            version="unknown",
        )
