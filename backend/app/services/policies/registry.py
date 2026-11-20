"""
app/services/policies/registry.py

Responsibility: Registry for discovering, loading, and caching versioned policy sets.
Exposes get_active_policy_set() which loads the version configured in configs/app.yaml.
"""

from __future__ import annotations

import functools
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import ConfigError
from app.services.policies.loader import load_policy_file
from app.services.policies.models import PolicySet

# Root directory containing versioned policy YAML files (backend/policies/)
_POLICIES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "policies"


@functools.lru_cache(maxsize=16)
def get_policy_set(version: str) -> PolicySet:
    """
    Retrieve and cache a specific policy set version from backend/policies/{version}.yaml.
    """
    policy_path = _POLICIES_DIR / f"{version}.yaml"
    if not policy_path.exists():
        raise ConfigError(
            f"Policy version '{version}' not found at {policy_path}. "
            f"Ensure {version}.yaml exists in backend/policies/."
        )
    return load_policy_file(policy_path)


def get_active_policy_set() -> PolicySet:
    """
    Load the active policy set as specified by app configuration (configs/app.yaml: policy_version).
    """
    settings = get_settings()
    version = settings.app.policy_version
    return get_policy_set(version)


def clear_policy_cache() -> None:
    """Clear cached policy sets (primarily for unit/integration tests)."""
    get_policy_set.cache_clear()
