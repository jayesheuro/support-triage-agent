"""
app/core/prompts.py

Responsibility: Load, cache, and format versioned prompt templates from backend/prompts/*.yaml.
Ensures prompt text is never hardcoded inline within agent nodes.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.core.errors import ConfigError

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_PROMPTS_DIR = _BACKEND_ROOT / "prompts"


class PromptTemplate(BaseModel):
    """Schema for versioned YAML prompt files."""

    name: str
    version: int
    changelog: str = ""
    template: str

    def format(self, **kwargs: Any) -> str:
        """Format the prompt template with dynamic arguments."""
        try:
            return self.template.format(**kwargs)
        except KeyError as exc:
            raise ConfigError(
                f"Missing required variable {exc} when formatting prompt '{self.name}' v{self.version}"
            ) from exc


@functools.lru_cache(maxsize=32)
def load_prompt(name: str) -> PromptTemplate:
    """
    Load a versioned prompt template by name from backend/prompts/<name>.yaml.
    Results are cached in memory.
    """
    prompt_file = _PROMPTS_DIR / f"{name}.yaml"
    if not prompt_file.exists():
        raise ConfigError(f"Prompt template file not found: {prompt_file}")

    try:
        with open(prompt_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return PromptTemplate.model_validate(data)
    except Exception as exc:
        raise ConfigError(f"Failed to load or validate prompt template '{name}': {exc}") from exc
