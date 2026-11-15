"""
app/services/providers/bedrock.py

Responsibility: AWS Bedrock LLM and embedder stub. boto3 is imported lazily
so the absence of the package doesn't break the app when provider != bedrock.

STATUS: Untested stub — written for completeness. Full implementation deferred.
The embedder raises NotImplementedError; the LLM makes a real boto3 call but
is not covered by integration tests.
"""

from __future__ import annotations

import json

from app.core.errors import ProviderAuthError, ProviderError
from app.core.logging import get_logger
from app.services.providers.base import BaseEmbedder, BaseLLM, LLMResponse

logger = get_logger(__name__)


class BedrockLLM(BaseLLM):
    """LLM provider backed by AWS Bedrock (Claude via boto3). UNTESTED STUB."""

    def __init__(
        self,
        region: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        # Lazy import — boto3 is not in pyproject.toml (Phase 9 concern)
        try:
            import boto3  # type: ignore[import]
        except ImportError as exc:
            raise ProviderError(
                "boto3 is required for the Bedrock provider. "
                "Install it: uv add boto3"
            ) from exc

        kwargs: dict[str, object] = {"region_name": region}
        if aws_access_key_id:
            kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            kwargs["aws_secret_access_key"] = aws_secret_access_key

        self._client = boto3.client("bedrock-runtime", **kwargs)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds

    def _generate_impl(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> LLMResponse:
        # Claude on Bedrock uses the Messages API
        system_text = ""
        bedrock_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                bedrock_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

        body: dict[str, object] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(kwargs.get("max_tokens", self._max_tokens)),
            "temperature": float(kwargs.get("temperature", self._temperature)),
            "messages": bedrock_messages,
        }
        if system_text:
            body["system"] = system_text

        try:
            response = self._client.invoke_model(
                modelId=self._model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        except Exception as exc:
            msg = str(exc)
            if "AccessDenied" in msg or "UnrecognizedClientException" in msg:
                raise ProviderAuthError(
                    "Bedrock auth failed — check AWS credentials in .env"
                ) from exc
            raise ProviderError(f"Bedrock error: {msg}") from exc

        result = json.loads(response["body"].read())
        text = result.get("content", [{}])[0].get("text", "")
        usage = result.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        return LLMResponse(
            text=text,
            model_name=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


class BedrockEmbedder(BaseEmbedder):
    """Embedder backed by AWS Bedrock Titan. UNTESTED STUB."""

    def __init__(
        self,
        region: str,
        model: str,
        dimension: int,
        timeout_seconds: int,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        try:
            import boto3  # type: ignore[import]
        except ImportError as exc:
            raise ProviderError("boto3 required for Bedrock provider.") from exc

        kwargs: dict[str, object] = {"region_name": region}
        if aws_access_key_id:
            kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            kwargs["aws_secret_access_key"] = aws_secret_access_key

        self._client = boto3.client("bedrock-runtime", **kwargs)
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_documents_impl(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_query_impl(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text})
        try:
            response = self._client.invoke_model(
                modelId=self._model,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
        except Exception as exc:
            raise ProviderError(f"Bedrock embed error: {exc}") from exc
        result = json.loads(response["body"].read())
        return result.get("embedding", [])
