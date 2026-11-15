"""
app/services/providers/factory.py

Responsibility: Read configs/app.yaml → instantiate the correct LLM + Embedder
pair. Zero code changes needed to swap providers — only configs/app.yaml changes.
Returns a cached (provider_name, BaseLLM, BaseEmbedder) tuple.
"""

from __future__ import annotations

import functools

from app.core.config import Settings, get_settings
from app.core.errors import ConfigError
from app.core.logging import get_logger
from app.services.providers.base import BaseEmbedder, BaseLLM

logger = get_logger(__name__)


def _build_ollama(settings: Settings) -> tuple[BaseLLM, BaseEmbedder]:
    from app.services.providers.ollama import OllamaEmbedder, OllamaLLM

    cfg = settings.provider
    llm = OllamaLLM(
        base_url=cfg.base_url or "http://localhost:11434",
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        timeout_seconds=cfg.llm.timeout_seconds,
    )
    embedder = OllamaEmbedder(
        base_url=cfg.base_url or "http://localhost:11434",
        model=cfg.embedder.model,
        dimension=cfg.embedder.dimension,
        timeout_seconds=cfg.embedder.timeout_seconds,
    )
    return llm, embedder


def _build_gemini(settings: Settings) -> tuple[BaseLLM, BaseEmbedder]:
    from app.services.providers.gemini import GeminiEmbedder, GeminiLLM

    api_key = settings.require_google_api_key()
    cfg = settings.provider
    llm = GeminiLLM(
        api_key=api_key,
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        timeout_seconds=cfg.llm.timeout_seconds,
    )
    embedder = GeminiEmbedder(
        api_key=api_key,
        model=cfg.embedder.model,
        dimension=cfg.embedder.dimension,
        timeout_seconds=cfg.embedder.timeout_seconds,
    )
    return llm, embedder


def _build_bedrock(settings: Settings) -> tuple[BaseLLM, BaseEmbedder]:
    from app.services.providers.bedrock import BedrockEmbedder, BedrockLLM

    key_id, secret = settings.require_aws_credentials()
    cfg = settings.provider
    region = cfg.region or settings.aws_default_region
    llm = BedrockLLM(
        region=region,
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        timeout_seconds=cfg.llm.timeout_seconds,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
    )
    embedder = BedrockEmbedder(
        region=region,
        model=cfg.embedder.model,
        dimension=cfg.embedder.dimension,
        timeout_seconds=cfg.embedder.timeout_seconds,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
    )
    return llm, embedder


_BUILDERS = {
    "ollama": _build_ollama,
    "gemini": _build_gemini,
    "bedrock": _build_bedrock,
}

_cached_providers: tuple[str, BaseLLM, BaseEmbedder] | None = None


def get_providers(settings: Settings | None = None) -> tuple[str, BaseLLM, BaseEmbedder]:
    """
    Return (provider_name, llm, embedder). Cached after first call when settings is None.

    Args:
        settings: Override for testing (default: get_settings()).

    Returns:
        Tuple of (provider_name, BaseLLM instance, BaseEmbedder instance).

    Raises:
        ConfigError: If the configured provider name is unknown.
    """
    global _cached_providers
    if settings is None and _cached_providers is not None:
        return _cached_providers

    resolved_settings = settings if settings is not None else get_settings()
    provider_name = resolved_settings.app.provider
    builder = _BUILDERS.get(provider_name)
    if builder is None:
        raise ConfigError(
            f"Unknown provider '{provider_name}'. "
            f"Valid options: {list(_BUILDERS.keys())}"
        )

    logger.info("Initializing provider", extra={"provider": provider_name})
    llm, embedder = builder(resolved_settings)
    result = (provider_name, llm, embedder)
    if settings is None:
        _cached_providers = result
    return result


def clear_provider_cache() -> None:
    """Clear the cached provider instance."""
    global _cached_providers
    _cached_providers = None


get_providers.cache_clear = clear_provider_cache  # type: ignore[attr-defined]


def get_llm(settings: Settings | None = None) -> BaseLLM:
    """Convenience accessor — returns just the LLM instance."""
    _, llm, _ = get_providers(settings)
    return llm


def get_embedder(settings: Settings | None = None) -> BaseEmbedder:
    """Convenience accessor — returns just the Embedder instance."""
    _, _, embedder = get_providers(settings)
    return embedder
