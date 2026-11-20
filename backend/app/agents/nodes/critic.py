"""
app/agents/nodes/critic.py

Responsibility: Critic agent node for policy compliance and safety verification.
Inspects drafted customer responses against the active versioned policy set.
Uses structured output parsing with retry feedback to detect violations, validate
against known policy IDs, and compute actionable verdicts (pass, revise, escalate).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.agents.state import AgentState
from app.agents.structured import call_structured
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.prompts import load_prompt
from app.services.policies.models import PolicySet
from app.services.policies.registry import get_active_policy_set
from app.services.providers.base import BaseLLM
from app.services.providers.factory import get_providers

logger = get_logger(__name__)


class CriticViolation(BaseModel):
    """Structured representation of a single policy violation."""

    policy_id: str = Field(..., description="The unique ID of the violated policy rule (e.g. POL-001)")
    severity: Literal["critical", "major", "minor"] = Field(
        ...,
        description="Severity level matching the rule definition",
    )
    quoted_span: str = Field(
        ...,
        description="Exact quote from the draft demonstrating violation, or indicator of missing requirement",
    )
    reason: str = Field(..., description="Explanation of why the span violates the rule")


class CriticOutput(BaseModel):
    """Raw structured output schema expected from the Critic LLM."""

    verdict: Literal["pass", "revise", "escalate"] = Field(
        ...,
        description="Overall verdict based on severity rules",
    )
    violations: list[CriticViolation] = Field(
        default_factory=list,
        description="List of detected policy violations",
    )
    overall_reasoning: str = Field(..., description="Concise rationale for the verdict")


def determine_verdict(
    violations: list[CriticViolation],
    revisions_used: int = 0,
    max_revision_loops: int = 1,
) -> str:
    """
    Pure function to determine canonical critic verdict from violations and revision budget:
    - Any 'critical' violation -> 'escalate' (cannot auto-send or automated revision).
    - Any 'major' violation -> 'revise' if revisions_used < max_revision_loops else 'escalate'.
    - Only 'minor' or zero violations -> 'pass'.
    """
    severities = [v.severity for v in violations]

    if "critical" in severities:
        return "escalate"

    if "major" in severities:
        if revisions_used < max_revision_loops:
            return "revise"
        return "escalate"

    return "pass"


def format_policy_rules_block(policy_set: PolicySet) -> str:
    """Format active policy rules into a readable block for the LLM prompt."""
    blocks: list[str] = []
    for rule in policy_set.rules:
        violating_str = "\n".join(f"      - {ex}" for ex in rule.examples_violating)
        compliant_str = "\n".join(f"      - {ex}" for ex in rule.examples_compliant)

        rule_text = (
            f"Rule {rule.id}: {rule.title} [Severity: {rule.severity.upper()}]\n"
            f"  Description: {rule.description}\n"
            f"  Violating Examples:\n{violating_str}\n"
            f"  Compliant Examples:\n{compliant_str}"
        )
        blocks.append(rule_text)

    return "\n\n".join(blocks)


def get_critic_schema(valid_policy_ids: set[str]) -> type[CriticOutput]:
    """
    Generate a dynamic Pydantic schema enforcing that all violation policy_ids
    exist in the active policy set. Unknown IDs trigger validation errors,
    which feed into call_structured's retry loop.
    """

    class ValidatedCriticOutput(CriticOutput):
        @model_validator(mode="after")
        def validate_policy_ids(self) -> ValidatedCriticOutput:
            for v in self.violations:
                if v.policy_id not in valid_policy_ids:
                    raise ValueError(
                        f"Unknown policy_id '{v.policy_id}'. "
                        f"Must be one of active policy IDs: {sorted(list(valid_policy_ids))}"
                    )
            return self

    return ValidatedCriticOutput


def critic_node(
    state: AgentState,
    llm: BaseLLM | None = None,
    policy_set: PolicySet | None = None,
) -> dict[str, Any]:
    """
    Evaluate candidate draft against the active policy set.
    Outputs structured verdict and violations, enforcing call budget and validation retries.
    """
    ticket_id = state.get("ticket_id", "unknown")
    llm_calls_used = state.get("llm_calls_used", 0)
    max_llm_calls = state.get("max_llm_calls", 6)
    revisions_used = state.get("revisions_used", 0)
    draft_body = state.get("draft_body")

    # If there is no draft (e.g. low confidence or kb gap), critic cannot review
    if not draft_body:
        logger.info(
            "Critic node skipped: no draft_body present in state",
            extra={"ticket_id": ticket_id},
        )
        return {
            "critic_verdict": "escalate",
            "critic_violations": [],
            "critic_reasoning": "Skipped because no draft body was produced.",
        }

    # 1. Budget enforcement check
    if llm_calls_used >= max_llm_calls:
        logger.error(
            "LLM call budget exceeded before critic node",
            extra={"ticket_id": ticket_id, "used": llm_calls_used, "max": max_llm_calls},
        )
        return {
            "status": "error",
            "critic_verdict": "escalate",
            "error": f"LLM call budget exceeded ({llm_calls_used}/{max_llm_calls})",
        }

    # 2. Resolve active LLM provider and policy set
    if llm is None:
        _, llm, _ = get_providers()

    if policy_set is None:
        policy_set = get_active_policy_set()

    settings = get_settings()
    max_retries = settings.app.max_structured_output_retries
    max_revision_loops = settings.app.max_revision_loops

    # 3. Format prompt
    policy_rules_block = format_policy_rules_block(policy_set)
    prompt_template = load_prompt("critic")
    prompt_text = prompt_template.format(
        policy_rules_block=policy_rules_block,
        customer_name=state.get("customer_name", "Customer"),
        channel=state.get("channel", "web"),
        category=state.get("category", "general"),
        priority=state.get("priority", "P2"),
        sentiment=state.get("sentiment", "neutral"),
        churn_risk=state.get("churn_risk", "low"),
        ticket_subject=state.get("subject", ""),
        ticket_body=state.get("body", ""),
        draft_body=draft_body,
    )

    # 4. Call structured LLM with retry loop and schema validation
    schema = get_critic_schema(policy_set.rule_ids)

    try:
        output, attempts = call_structured(
            llm=llm,
            prompt=prompt_text,
            schema=schema,
            max_retries=max_retries,
            return_metadata=True,
            temperature=0.0,
        )
        llm_calls_used += attempts
    except Exception as exc:
        logger.error(
            "Critic node structured call failed after retries",
            extra={"ticket_id": ticket_id, "error": str(exc)},
        )
        return {
            "critic_verdict": "escalate",
            "critic_violations": [],
            "critic_reasoning": f"Critic evaluation failed: {exc}",
            "llm_calls_used": llm_calls_used + max_retries + 1,
            "error": str(exc),
        }

    # 5. Enforce deterministic business logic on verdict mapping
    canonical_verdict = determine_verdict(
        violations=output.violations,
        revisions_used=revisions_used,
        max_revision_loops=max_revision_loops,
    )

    violations_data = [v.model_dump() for v in output.violations]

    logger.info(
        "Critic evaluation completed",
        extra={
            "ticket_id": ticket_id,
            "verdict": canonical_verdict,
            "violations_count": len(violations_data),
            "revisions_used": revisions_used,
            "llm_calls_used": llm_calls_used,
        },
    )

    return {
        "critic_verdict": canonical_verdict,
        "critic_violations": violations_data,
        "critic_reasoning": output.overall_reasoning,
        "llm_calls_used": llm_calls_used,
    }
