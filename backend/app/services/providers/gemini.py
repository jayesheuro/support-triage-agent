"""
app/services/providers/gemini.py

Responsibility: Gemini LLM and embedder implementations using the google-genai SDK
(package: google-genai >= 1.2.0). Current SDK API: client.models.generate_content()
and client.models.embed_content(). Requires GOOGLE_API_KEY in .env.
"""

from __future__ import annotations

from app.core.errors import ProviderAuthError, ProviderError, ProviderRateLimitError
from app.core.logging import get_logger
from app.services.providers.base import BaseEmbedder, BaseLLM, LLMResponse

logger = get_logger(__name__)


class GeminiLLM(BaseLLM):
    """LLM provider backed by Google Gemini (google-genai SDK)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
    ) -> None:
        # Import here so missing google-genai doesn't fail at module load
        from google import genai  # type: ignore[import]
        from google.genai import types  # type: ignore[import]

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds

    def _generate_impl(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> LLMResponse:
        from google.genai import types  # type: ignore[import]

        # Separate system prompt from conversation turns
        system_parts: list[str] = []
        contents: list[object] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content)],
                    )
                )
            elif role == "assistant":
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=content)],
                    )
                )

        config = types.GenerateContentConfig(
            system_instruction="\n\n".join(system_parts) if system_parts else None,
            temperature=float(kwargs.get("temperature", self._temperature)),
            max_output_tokens=int(kwargs.get("max_tokens", self._max_tokens)),
        )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            self._translate_error(exc)

        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

        return LLMResponse(
            text=text,
            model_name=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _translate_error(self, exc: Exception) -> None:
        """Map Gemini SDK exceptions to our hierarchy and re-raise."""
        msg = str(exc)
        if "API_KEY" in msg or "403" in msg or "401" in msg:
            raise ProviderAuthError(
                "Gemini authentication failed — check GOOGLE_API_KEY in .env"
            ) from exc
        if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
            raise ProviderRateLimitError(f"Gemini rate limit: {msg}") from exc
        raise ProviderError(f"Gemini error: {msg}") from exc


class GeminiEmbedder(BaseEmbedder):
    """Embedder backed by Google Gemini text-embedding-004."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dimension: int,
        timeout_seconds: int,
    ) -> None:
        from google import genai  # type: ignore[import]

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimension = dimension
        self._timeout = timeout_seconds

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_documents_impl(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            results.append(self._embed_one(text))
        return results

    def _embed_query_impl(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        try:
            response = self._client.models.embed_content(
                model=self._model,
                contents=text,
            )
        except Exception as exc:
            msg = str(exc)
            if "API_KEY" in msg or "403" in msg or "401" in msg:
                raise ProviderAuthError(
                    "Gemini auth failed on embed — check GOOGLE_API_KEY"
                ) from exc
            raise ProviderError(f"Gemini embed error: {msg}") from exc

        # SDK returns EmbedContentResponse with .embeddings list
        embeddings = getattr(response, "embeddings", None)
        if embeddings and len(embeddings) > 0:
            values = getattr(embeddings[0], "values", None)
            if values:
                return list(values)
        raise ProviderError(
            f"Gemini embed returned unexpected response format: {response}"
        )
