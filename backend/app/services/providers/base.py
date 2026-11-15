"""
app/services/providers/base.py

Responsibility: Define the abstract base classes for LLM and Embedder providers,
the shared LLMResponse data model, and the tenacity retry/backoff decorator that
all concrete providers inherit. Retry and timeout logic lives here — NOT at call
sites — so every provider automatically gets robust failure handling.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import httpx
import tenacity
from pydantic import BaseModel

from app.core.errors import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared response model
# ---------------------------------------------------------------------------

class LLMResponse(BaseModel):
    """Structured result from a generate() call."""

    text: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Retry predicate helpers
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors that should trigger a retry."""
    return isinstance(exc, (ProviderTimeoutError, ProviderRateLimitError, ProviderUnavailableError))


def _before_retry_log(retry_state: tenacity.RetryCallState) -> None:
    logger.warning(
        "Provider retry",
        extra={
            "attempt": retry_state.attempt_number,
            "error": str(retry_state.outcome.exception()) if retry_state.outcome else None,
        },
    )


# Shared tenacity retry decorator — applied by concrete implementations
# 3 attempts, exponential backoff: 1 s → 2 s → 4 s, max 10 s wait
_provider_retry = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
    stop=tenacity.stop_after_attempt(3),
    before_sleep=_before_retry_log,
    reraise=True,
)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class BaseLLM(ABC):
    """
    Abstract LLM provider. Concrete implementations call _generate_impl()
    wrapped by the shared retry decorator.

    Message format: list of dicts with "role" ("system" | "user" | "assistant")
    and "content" (str). This is the same schema as OpenAI-compatible APIs and
    maps trivially to Gemini's Content objects.
    """

    def generate(self, messages: list[dict[str, str]], **kwargs: object) -> LLMResponse:
        """
        Send a chat-style prompt to the provider and return a structured response.

        Args:
            messages: Conversation turns, e.g.
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            **kwargs: Provider-specific overrides (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with text, token counts, and latency.

        Raises:
            ProviderError: On unrecoverable provider failures.
        """
        return self._generate_with_retry(messages, **kwargs)

    @_provider_retry
    def _generate_with_retry(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> LLMResponse:
        """Retry-wrapped call to the concrete implementation."""
        t0 = time.monotonic()
        try:
            result = self._generate_impl(messages, **kwargs)
            result.latency_ms = (time.monotonic() - t0) * 1000
            logger.debug(
                "LLM generate OK",
                extra={
                    "model": result.model_name,
                    "total_tokens": result.total_tokens,
                    "latency_ms": round(result.latency_ms, 1),
                },
            )
            return result
        except (ProviderTimeoutError, ProviderRateLimitError, ProviderUnavailableError):
            raise  # let tenacity retry
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Cannot connect to provider — is it running? ({exc})"
            ) from exc
        # NOTE: Non-AppError exceptions (ValueError, RuntimeError, etc.) are NOT
        # caught here — they propagate to the agent node's structured-output retry
        # loop. The base layer handles transient network failures only.

    @abstractmethod
    def _generate_impl(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> LLMResponse:
        """Concrete providers implement this. No retry logic here."""
        ...


class BaseEmbedder(ABC):
    """
    Abstract embedder provider. Returns fixed-dimension float vectors.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents. Returns list of float vectors."""
        return self._embed_documents_with_retry(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Returns a float vector."""
        return self._embed_query_with_retry(text)

    @_provider_retry
    def _embed_documents_with_retry(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._embed_documents_impl(texts)
        except (ProviderTimeoutError, ProviderRateLimitError, ProviderUnavailableError):
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Embed request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Cannot connect to embedder — is it running? ({exc})"
            ) from exc
        except Exception as exc:
            raise ProviderError(f"Unexpected embedder error: {exc}") from exc

    @_provider_retry
    def _embed_query_with_retry(self, text: str) -> list[float]:
        try:
            return self._embed_query_impl(text)
        except (ProviderTimeoutError, ProviderRateLimitError, ProviderUnavailableError):
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Embed request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Cannot connect to embedder — is it running? ({exc})"
            ) from exc
        except Exception as exc:
            raise ProviderError(f"Unexpected embedder error: {exc}") from exc

    @abstractmethod
    def _embed_documents_impl(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def _embed_query_impl(self, text: str) -> list[float]:
        ...
