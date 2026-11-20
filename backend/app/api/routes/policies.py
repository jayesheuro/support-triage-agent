"""
app/api/routes/policies.py

Responsibility: REST API route for reading the active compliance policy set.
The Phase 8 UI reads this to display rules, severity tiers, and examples.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.models.policies import PolicyRuleResponse, PolicySetResponse
from app.services.policies.registry import get_active_policy_set

router = APIRouter(tags=["policies"])


@router.get("/policies", response_model=PolicySetResponse)
def get_policies() -> PolicySetResponse:
    """
    Get the currently active policy set and version.
    """
    policy_set = get_active_policy_set()
    return PolicySetResponse(
        version=policy_set.version,
        changelog=policy_set.changelog,
        rules=[
            PolicyRuleResponse(
                id=rule.id,
                title=rule.title,
                description=rule.description,
                severity=rule.severity,
                examples_violating=rule.examples_violating,
                examples_compliant=rule.examples_compliant,
            )
            for rule in policy_set.rules
        ],
    )
