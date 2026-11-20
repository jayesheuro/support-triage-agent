"""
app/services/policies/models.py

Responsibility: Strongly typed Pydantic models for the versioned policy set and rules.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PolicyRule(BaseModel):
    """A single compliance or safety rule in the policy set."""

    id: str = Field(..., description="Unique policy identifier (e.g. POL-001)")
    title: str = Field(..., description="Short title describing the rule")
    description: str = Field(..., description="Detailed normative requirement")
    severity: Literal["critical", "major", "minor"] = Field(
        ...,
        description="Severity level: critical (escalate), major (revisable), minor (advisory)",
    )
    examples_violating: list[str] = Field(
        default_factory=list,
        description="Examples of non-compliant draft text",
    )
    examples_compliant: list[str] = Field(
        default_factory=list,
        description="Examples of compliant draft text",
    )


class PolicySet(BaseModel):
    """Complete versioned policy artifact containing compliance rules."""

    version: str = Field(..., description="Policy version identifier (e.g. shoplite_v1)")
    changelog: str = Field(..., description="Summary of changes in this policy version")
    rules: list[PolicyRule] = Field(..., description="Ordered list of policy rules")

    @property
    def rule_ids(self) -> set[str]:
        """Return the set of all rule IDs present in this policy set."""
        return {rule.id for rule in self.rules}

    def get_rule(self, rule_id: str) -> PolicyRule | None:
        """Find a rule by its ID, or return None if not found."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None
