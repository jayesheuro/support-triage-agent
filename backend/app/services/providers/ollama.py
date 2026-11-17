"""
app/services/providers/ollama.py

Responsibility: Ollama local LLM and embedder implementations. Communicates
with the Ollama REST API over HTTP using httpx (synchronous). Default provider.

Ollama must be running: https://ollama.com
Required models:
  ollama pull llama3.2:latest
  ollama pull nomic-embed-text
"""

from __future__ import annotations

import httpx

from app.core.errors import ProviderAuthError, ProviderError, ProviderUnavailableError
from app.core.logging import get_logger
from app.services.providers.base import BaseEmbedder, BaseLLM, LLMResponse

logger = get_logger(__name__)

# Ollama API uses these role names (same as OpenAI-compatible)
_ROLE_MAP = {"system": "system", "user": "user", "assistant": "assistant"}


class OllamaLLM(BaseLLM):
    """LLM provider backed by a local Ollama instance."""

    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int, timeout_seconds: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = httpx.Timeout(timeout_seconds)

    def _generate_impl(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self._temperature),
                "num_predict": kwargs.get("max_tokens", self._max_tokens),
            },
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base_url}/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                "Ollama is not running. Start it with: ollama serve"
            ) from exc

        if resp.status_code == 401:
            raise ProviderAuthError("Ollama returned 401 — unexpected auth error.")
        if not resp.is_success:
            raise ProviderError(
                f"Ollama /api/chat error {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        content = data.get("message", {}).get("content", "")
        # Token usage fields (present in Ollama >= 0.1.14)
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return LLMResponse(
            text=content,
            model_name=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


class OllamaEmbedder(BaseEmbedder):
    """Embedder backed by a local Ollama instance using nomic-embed-text."""

    def __init__(self, base_url: str, model: str, dimension: int, timeout_seconds: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = dimension
        self._timeout = httpx.Timeout(timeout_seconds)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_single(self, text: str) -> list[float]:
        """Call /api/embed for a single text string."""
        payload = {"model": self._model, "input": text}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base_url}/api/embed", json=payload)
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                "Ollama is not running. Start it with: ollama serve"
            ) from exc

        if not resp.is_success:
            raise ProviderError(
                f"Ollama /api/embed error {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        # Ollama /api/embed returns {"embeddings": [[...]], ...}
        embeddings = data.get("embeddings")
        if embeddings and isinstance(embeddings, list):
            return embeddings[0]
        # Fallback to old /api/embeddings format
        embedding = data.get("embedding")
        if embedding:
            return embedding
        raise ProviderError(f"Ollama embed returned unexpected format: {list(data.keys())}")

    def _embed_documents_impl(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self._model, "input": texts}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base_url}/api/embed", json=payload)
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                "Ollama is not running. Start it with: ollama serve"
            ) from exc

        if not resp.is_success:
            raise ProviderError(
                f"Ollama /api/embed error {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings and isinstance(embeddings, list):
            return embeddings
        # Fallback for older Ollama versions
        return [self._embed_single(t) for t in texts]

    def _embed_query_impl(self, text: str) -> list[float]:
        return self._embed_single(text)
