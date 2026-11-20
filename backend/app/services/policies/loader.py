"""
app/services/policies/loader.py

Responsibility: Parse and validate policy set YAML files into PolicySet models.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from app.core.errors import ConfigError
from app.services.policies.models import PolicySet


def load_policy_file(file_path: Path | str) -> PolicySet:
    """
    Load and validate a policy YAML file from disk.

    Raises:
        ConfigError: If the file does not exist or fails schema validation.
    """
    path = Path(file_path)
    if not path.is_file():
        raise ConfigError(f"Policy file not found at: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except Exception as e:
        raise ConfigError(f"Failed to parse policy YAML at {path}: {e}") from e

    if not isinstance(raw_data, dict):
        raise ConfigError(f"Policy file at {path} must contain a top-level mapping")

    try:
        return PolicySet.model_validate(raw_data)
    except ValidationError as e:
        raise ConfigError(f"Policy schema validation failed for {path}: {e}") from e
