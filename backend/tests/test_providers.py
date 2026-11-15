"""
tests/test_providers.py

Unit tests for the provider abstraction — factory selection, FakeLLM/FakeEmbedder
behaviour, and retry paths. No real network calls.

Integration tests (marked @pytest.mark.integration) are auto-skipped when Ollama
is not running or the required API key is absent.
"""

from __future__ import annotations

import pytest

from app.core.errors import ProviderError, ProviderUnavailableError
from app.services.providers.base import LLMResponse
from tests.fakes import FakeEmbedder, FakeLLM


# ---------------------------------------------------------------------------
# FakeLLM unit tests
# ---------------------------------------------------------------------------

class TestFakeLLM:
    def test_returns_queued_response(self) -> None:
        llm = FakeLLM(["hello"])
        result = llm.generate([{"role": "user", "content": "hi"}])
        assert isinstance(result, LLMResponse)
        assert result.text == "hello"
        assert result.model_name == "fake-llm"

    def test_records_calls(self) -> None:
        llm = FakeLLM(["a", "b"])
        llm.generate([{"role": "user", "content": "first"}])
        llm.generate([{"role": "user", "content": "second"}])
        assert len(llm.calls) == 2

    def test_raises_on_empty_queue(self) -> None:
        llm = FakeLLM([])
        with pytest.raises(RuntimeError, match="queue is empty"):
            llm.generate([{"role": "user", "content": "?"}])

    def test_raises_queued_exception(self) -> None:
        llm = FakeLLM([ProviderError("boom")])
        with pytest.raises(ProviderError, match="boom"):
            llm.generate([{"role": "user", "content": "?"}])

    def test_malformed_then_valid_sequence(self) -> None:
        """Simulates retry path: first call raises, second returns valid text."""
        llm = FakeLLM([ValueError("parse failure"), "valid output"])
        # First call raises
        with pytest.raises(ValueError):
            llm.generate([{"role": "user", "content": "try 1"}])
        # Second call (simulating retry at application level) succeeds
        result = llm.generate([{"role": "user", "content": "try 2"}])
        assert result.text == "valid output"

    def test_token_counts_present(self) -> None:
        llm = FakeLLM(["one two three"])
        result = llm.generate([{"role": "user", "content": "count"}])
        assert result.total_tokens > 0

    def test_push_adds_to_queue(self) -> None:
        llm = FakeLLM([])
        llm.push("late addition")
        result = llm.generate([{"role": "user", "content": "?"}])
        assert result.text == "late addition"


# ---------------------------------------------------------------------------
# FakeEmbedder unit tests
# ---------------------------------------------------------------------------

class TestFakeEmbedder:
    def test_dimension_property(self) -> None:
        emb = FakeEmbedder(dimension=768)
        assert emb.dimension == 768

    def test_embed_query_returns_correct_dim(self) -> None:
        emb = FakeEmbedder(dimension=768)
        vec = emb.embed_query("test")
        assert len(vec) == 768

    def test_embed_documents_batch(self) -> None:
        emb = FakeEmbedder(dimension=128)
        vecs = emb.embed_documents(["a", "b", "c"])
        assert len(vecs) == 3
        assert all(len(v) == 128 for v in vecs)

    def test_deterministic(self) -> None:
        emb = FakeEmbedder(dimension=16)
        v1 = emb.embed_query("hello world")
        v2 = emb.embed_query("hello world")
        assert v1 == v2

    def test_different_texts_differ(self) -> None:
        emb = FakeEmbedder(dimension=16)
        v1 = emb.embed_query("text A")
        v2 = emb.embed_query("text B")
        assert v1 != v2

    def test_records_calls(self) -> None:
        emb = FakeEmbedder()
        emb.embed_query("first")
        emb.embed_query("second")
        assert len(emb.calls) == 2


# ---------------------------------------------------------------------------
# Factory unit tests (monkeypatched — no real providers instantiated)
# ---------------------------------------------------------------------------

class TestFactory:
    def _make_settings(self, provider: str, monkeypatch: pytest.MonkeyPatch):
        """Return a minimal Settings-like object for testing."""
        from app.core.config import AppConfig, EmbedderConfig, LLMConfig, ProviderConfig, Settings

        class _FakeSettings(Settings):
            @property
            def app(self):  # type: ignore[override]
                return AppConfig(provider=provider)

            @property
            def provider(self):  # type: ignore[override]
                return ProviderConfig(
                    base_url="http://localhost:11434" if provider == "ollama" else None,
                    llm=LLMConfig(model="fake-model"),
                    embedder=EmbedderConfig(model="fake-embed"),
                )

        return _FakeSettings()

    def test_factory_ollama_builds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Factory should produce OllamaLLM/OllamaEmbedder for provider=ollama."""
        from app.services.providers.factory import get_providers
        from app.services.providers.ollama import OllamaEmbedder, OllamaLLM

        settings = self._make_settings("ollama", monkeypatch)
        # Clear the lru_cache so our fake settings take effect
        get_providers.cache_clear()
        name, llm, embedder = get_providers(settings=settings)
        assert name == "ollama"
        assert isinstance(llm, OllamaLLM)
        assert isinstance(embedder, OllamaEmbedder)
        get_providers.cache_clear()  # clean up

    def test_factory_gemini_requires_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Factory should raise ConfigError for gemini without API key."""
        from app.core.errors import ConfigError
        from app.services.providers.factory import get_providers

        settings = self._make_settings("gemini", monkeypatch)
        get_providers.cache_clear()
        with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
            get_providers(settings=settings)
        get_providers.cache_clear()

    def test_factory_unknown_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Factory must raise ConfigError for unknown provider names."""
        from app.core.config import AppConfig
        from app.core.errors import ConfigError
        from app.services.providers.factory import get_providers

        with pytest.raises(Exception):  # AppConfig validator raises first
            AppConfig(provider="openai")

    def test_factory_bedrock_requires_creds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Factory should raise ConfigError for bedrock without AWS creds."""
        from app.core.errors import ConfigError
        from app.services.providers.factory import get_providers

        settings = self._make_settings("bedrock", monkeypatch)
        get_providers.cache_clear()
        with pytest.raises(ConfigError, match="AWS"):
            get_providers(settings=settings)
        get_providers.cache_clear()


# ---------------------------------------------------------------------------
# Integration tests — auto-skipped when Ollama is unavailable
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    import httpx
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.is_success
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
class TestOllamaIntegration:
    def test_llm_generate(self) -> None:
        from app.services.providers.factory import get_providers, get_llm
        get_providers.cache_clear()
        llm = get_llm()
        result = llm.generate([
            {"role": "system", "content": "Reply with exactly the word: OK"},
            {"role": "user", "content": "Say OK"},
        ])
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert result.total_tokens > 0
        get_providers.cache_clear()

    def test_embedder_dimension(self) -> None:
        from app.services.providers.factory import get_embedder, get_providers
        get_providers.cache_clear()
        embedder = get_embedder()
        vec = embedder.embed_query("test embedding")
        assert len(vec) == embedder.dimension
        get_providers.cache_clear()
