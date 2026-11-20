"""
app/core/errors.py

Responsibility: Define the application exception hierarchy and FastAPI exception
handlers that convert errors to structured JSON responses. All custom exceptions
descend from AppError so callers can catch broadly or narrowly.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class AppError(Exception):
    """Base class for all application-defined exceptions."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class ConfigError(AppError):
    """Raised when configuration is missing, invalid, or inconsistent."""


class ProviderError(AppError):
    """Base class for LLM / embedder provider failures."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request exceeds its configured timeout."""


class ProviderRateLimitError(ProviderError):
    """Raised when the provider returns a rate-limit or quota error."""


class ProviderAuthError(ProviderError):
    """Raised when provider authentication fails (bad key, missing creds)."""


class ProviderUnavailableError(ProviderError):
    """Raised when the provider endpoint is unreachable (e.g. Ollama not running)."""


class ParseError(AppError):
    """Raised when structured LLM output cannot be parsed after max retries."""


class StructuredOutputError(AppError):
    """Raised when structured LLM output fails validation after max retries."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: str | None = None,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message, detail=last_error)
        self.attempts = attempts
        self.last_error = last_error
        self.raw_response = raw_response


class KBError(AppError):
    """Raised for knowledge-base ingestion or retrieval failures."""


class KBDimensionMismatchError(KBError):
    """Raised when existing vector store dimension does not match active embedder dimension."""


class NotFoundError(AppError):
    """Raised when a requested resource does not exist."""


class ValidationError(AppError):
    """Raised for domain-level validation failures (distinct from Pydantic errors)."""


# ---------------------------------------------------------------------------
# HTTP status mapping
# ---------------------------------------------------------------------------

_STATUS_MAP: dict[type[AppError], int] = {
    ConfigError: 500,
    ProviderTimeoutError: 504,
    ProviderRateLimitError: 429,
    ProviderAuthError: 503,
    ProviderUnavailableError: 503,
    ProviderError: 502,
    ParseError: 422,
    StructuredOutputError: 422,
    KBDimensionMismatchError: 409,
    KBError: 500,
    NotFoundError: 404,
    ValidationError: 400,
    AppError: 500,
}


def _status_for(exc: AppError) -> int:
    for exc_type, status in _STATUS_MAP.items():
        if isinstance(exc, exc_type):
            return status
    return 500


# ---------------------------------------------------------------------------
# FastAPI exception handlers
# ---------------------------------------------------------------------------

def register_exception_handlers(app: FastAPI) -> None:
    """Attach all custom exception handlers to a FastAPI application instance."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=_status_for(exc),
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "detail": exc.detail,
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Log the full traceback at ERROR level; return a safe generic message.
        import logging
        logging.getLogger(__name__).exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred.",
                "detail": None,
            },
        )
