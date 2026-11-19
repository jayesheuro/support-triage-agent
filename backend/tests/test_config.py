"""
tests/test_config.py

Unit tests for the config loading and validation system. Uses temporary
YAML files so no real configs are required.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.core.config import AppConfig, ProviderConfig, TaxonomyConfig, _load_yaml
from app.core.errors import ConfigError


class TestAppConfig:
    def test_valid_provider_accepted(self) -> None:
        cfg = AppConfig(provider="ollama")
        assert cfg.provider == "ollama"

    def test_all_valid_providers(self) -> None:
        for p in ("ollama", "gemini", "bedrock"):
            cfg = AppConfig(provider=p)
            assert cfg.provider == p

    def test_invalid_provider_raises(self) -> None:
        with pytest.raises(Exception):  # Pydantic ValidationError
            AppConfig(provider="openai")

    def test_defaults(self) -> None:
        cfg = AppConfig(provider="ollama")
        assert cfg.max_llm_calls_per_ticket == 6
        assert cfg.auto_send_enabled is False
        assert cfg.confidence_threshold == 0.70

    def test_auto_send_default_false(self) -> None:
        """auto_send must default False — safety invariant."""
        cfg = AppConfig(provider="ollama")
        assert cfg.auto_send_enabled is False


class TestProviderConfig:
    def test_ollama_no_base_url_ok(self) -> None:
        from app.core.config import LLMConfig, EmbedderConfig
        cfg = ProviderConfig(
            llm=LLMConfig(model="llama3.2:latest"),
            embedder=EmbedderConfig(model="nomic-embed-text"),
        )
        assert cfg.base_url is None

    def test_ollama_with_base_url(self) -> None:
        from app.core.config import LLMConfig, EmbedderConfig
        cfg = ProviderConfig(
            base_url="http://localhost:11434",
            llm=LLMConfig(model="llama3.2:latest"),
            embedder=EmbedderConfig(model="nomic-embed-text"),
        )
        assert cfg.base_url == "http://localhost:11434"


class TestTaxonomyConfig:
    def test_defaults(self) -> None:
        cfg = TaxonomyConfig()
        assert isinstance(cfg.categories, list)
        assert cfg.confidence_threshold == 0.70

    def test_auto_send_allowlist_is_list(self) -> None:
        cfg = TaxonomyConfig(auto_send_allowlist=["order_status"])
        assert "order_status" in cfg.auto_send_allowlist

    def test_load_real_taxonomy_yaml(self) -> None:
        raw = _load_yaml("configs/taxonomy.yaml")
        cfg = TaxonomyConfig(**raw)
        assert "order_status" in cfg.category_names
        assert "refund_billing" in cfg.category_names
        assert "account_access" in cfg.never_auto_send
        assert cfg.routing["refund_billing"] == "billing"
        assert "P1" in cfg.priority_names
        assert cfg.confidence_threshold == 0.70


class TestLoadYaml:
    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            _load_yaml("configs/nonexistent_file_xyz.yaml")


class TestSettings:
    def test_settings_loads_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings should load fine with no .env present."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        from app.core.config import Settings
        s = Settings()
        assert s.google_api_key is None

    def test_require_google_api_key_raises_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        from app.core.config import Settings
        s = Settings()
        with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
            s.require_google_api_key()

    def test_require_aws_raises_without_creds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        from app.core.config import Settings
        s = Settings()
        with pytest.raises(ConfigError, match="AWS"):
            s.require_aws_credentials()
