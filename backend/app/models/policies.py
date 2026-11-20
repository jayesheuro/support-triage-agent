"""
app/models/policies.py

Responsibility: Pydantic request and response schemas for policy endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PolicyRuleResponse(BaseModel):
    """Single policy rule schema returned via the API."""

    id: str
    title: str
    description: str
    severity: str
    examples_violating: list[str] = Field(default_factory=list)
    examples_compliant: list[str] = Field(default_factory=list)


class PolicySetResponse(BaseModel):
    """Complete active policy set returned via GET /api/policies."""

    version: str
    changelog: str
    rules: list[PolicyRuleResponse]
