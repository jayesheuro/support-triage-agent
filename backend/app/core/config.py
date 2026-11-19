"""
app/core/config.py

Responsibility: Load and validate all application configuration from YAML files
and environment variables. Provides a single get_settings() accessor used
throughout the app. Secrets come from .env via Pydantic Settings; structural
config comes from configs/*.yaml via PyYAML.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Locate the backend root (the directory containing configs/)
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent  # backend/


def _load_yaml(relative_path: str) -> dict[str, Any]:
    """Load a YAML file relative to the backend root directory."""
    full_path = _BACKEND_ROOT / relative_path
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")
    with full_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Typed config models
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    """LLM-specific configuration for a provider."""

    model: str
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_seconds: int = 60


class EmbedderConfig(BaseModel):
    """Embedder-specific configuration for a provider."""

    model: str
    dimension: int = 768
    timeout_seconds: int = 30


class ProviderConfig(BaseModel):
    """Full provider configuration loaded from configs/providers/<name>.yaml."""

    base_url: str | None = None  # Ollama local endpoint; None for cloud providers
    llm: LLMConfig
    embedder: EmbedderConfig

    # Optional Bedrock-specific field
    region: str | None = None


class AppConfig(BaseModel):
    """Structural application settings from configs/app.yaml."""

    provider: str
    max_llm_calls_per_ticket: int = 6
    max_structured_output_retries: int = 2
    max_revision_loops: int = 1
    relevance_threshold: float = 0.50
    confidence_threshold: float = 0.70
    auto_send_enabled: bool = False
    version: str = "0.1.0"

    @field_validator("provider")
    @classmethod
    def provider_must_be_valid(cls, v: str) -> str:
        allowed = {"ollama", "gemini", "bedrock"}
        if v not in allowed:
            raise ValueError(f"provider must be one of {allowed}, got '{v}'")
        return v


class CategoryInfo(BaseModel):
    """Metadata for a ticket classification category."""

    description: str
    routing: str


class PriorityInfo(BaseModel):
    """Metadata and rules for a ticket priority level."""

    description: str
    rules: list[str] = Field(default_factory=list)


class TaxonomyConfig(BaseModel):
    """Taxonomy settings from configs/taxonomy.yaml."""

    categories: dict[str, CategoryInfo] | list[str] = Field(default_factory=list)
    priorities: dict[str, PriorityInfo] | list[str] = Field(default_factory=list)
    auto_send_allowlist: list[str] = Field(default_factory=list)
    never_auto_send: list[str] = Field(default_factory=list)
    confidence_threshold: float = 0.70

    @property
    def category_names(self) -> list[str]:
        if isinstance(self.categories, dict):
            return list(self.categories.keys())
        return list(self.categories)

    @property
    def priority_names(self) -> list[str]:
        if isinstance(self.priorities, dict):
            return list(self.priorities.keys())
        return list(self.priorities)

    @property
    def routing(self) -> dict[str, str]:
        if isinstance(self.categories, dict):
            return {k: v.routing for k, v in self.categories.items()}
        return {}


# ---------------------------------------------------------------------------
# Pydantic Settings — loads secrets from .env
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Top-level settings object. Secrets come from .env; YAML configs are loaded
    lazily via properties and cached. Use get_settings() to access a singleton.
    """

    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets — loaded from .env
    google_api_key: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_default_region: str = "us-east-1"

    # Optional override for log level
    log_level: str = "INFO"

    # ── Cached YAML config properties ───────────────────────────────────────

    @functools.cached_property
    def app(self) -> AppConfig:
        """Load and validate configs/app.yaml."""
        raw = _load_yaml("configs/app.yaml")
        return AppConfig(**raw)

    @functools.cached_property
    def provider(self) -> ProviderConfig:
        """Load configs/providers/<active_provider>.yaml."""
        provider_name = self.app.provider
        raw = _load_yaml(f"configs/providers/{provider_name}.yaml")
        return ProviderConfig(**raw)

    @functools.cached_property
    def taxonomy(self) -> TaxonomyConfig:
        """Load configs/taxonomy.yaml."""
        raw = _load_yaml("configs/taxonomy.yaml")
        return TaxonomyConfig(**raw)

    def require_google_api_key(self) -> str:
        """Return Google API key or raise ConfigError if absent."""
        from app.core.errors import ConfigError

        if not self.google_api_key:
            raise ConfigError(
                "GOOGLE_API_KEY is not set in .env — required when provider=gemini. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        return self.google_api_key

    def require_aws_credentials(self) -> tuple[str, str]:
        """Return (key_id, secret) or raise ConfigError if absent."""
        from app.core.errors import ConfigError

        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ConfigError(
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in .env "
                "when provider=bedrock."
            )
        return self.aws_access_key_id, self.aws_secret_access_key


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance. Cached after first call."""
    return Settings()
