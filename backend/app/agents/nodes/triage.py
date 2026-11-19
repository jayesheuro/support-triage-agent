"""
app/agents/nodes/triage.py

Responsibility: Triage agent node.
Classifies incoming support tickets into taxonomy categories, determines customer
sentiment and churn risk, computes confidence, and enforces priority escalation rules.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.agents.structured import call_structured
from app.core.config import get_settings
from app.core.errors import StructuredOutputError
from app.core.logging import get_logger
from app.core.prompts import load_prompt
from app.db.enums import TicketPriority, TicketStatus
from app.services.providers.base import BaseLLM
from app.services.providers.factory import get_providers

logger = get_logger(__name__)


class TriageOutput(BaseModel):
    """Structured output expected from the LLM triage classification."""

    category: str = Field(description="Taxonomy category key")
    priority: str = Field(description="Assigned priority: P1, P2, or P3")
    sentiment: str = Field(description="Customer sentiment: positive, neutral, negative, or angry")
    churn_risk: str = Field(description="Churn risk level: low, medium, or high")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Concise rationale for classification")


def _format_taxonomy_blocks() -> tuple[str, str]:
    """Format categories and priority rules from taxonomy config for prompt injection."""
    settings = get_settings()
    tax = settings.taxonomy

    cat_lines: list[str] = []
    if isinstance(tax.categories, dict):
        for name, info in tax.categories.items():
            cat_lines.append(f"- {name}: {info.description} (Routing: {info.routing})")
    else:
        for name in tax.categories:
            cat_lines.append(f"- {name}")
    categories_block = "\n".join(cat_lines)

    prio_lines: list[str] = []
    if isinstance(tax.priorities, dict):
        for name, pinfo in tax.priorities.items():
            rules_str = "; ".join(pinfo.rules) if pinfo.rules else "General priority level"
            prio_lines.append(f"- {name}: {pinfo.description} [Rules: {rules_str}]")
    else:
        for name in tax.priorities:
            prio_lines.append(f"- {name}")
    priorities_block = "\n".join(prio_lines)

    return categories_block, priorities_block


def triage_node(state: AgentState, llm: BaseLLM | None = None) -> dict[str, Any]:
    """
    Execute triage classification on the current ticket in graph state.
    Enforces LLM call budget and handles parse failures gracefully.
    """
    ticket_id = state.get("ticket_id", "unknown")
    llm_calls_used = state.get("llm_calls_used", 0)
    max_llm_calls = state.get("max_llm_calls", 6)

    # 1. Budget enforcement check
    if llm_calls_used >= max_llm_calls:
        logger.error(
            "LLM call budget exceeded before triage node",
            extra={"ticket_id": ticket_id, "used": llm_calls_used, "max": max_llm_calls},
        )
        return {
            "status": "error",
            "error": f"LLM call budget exceeded ({llm_calls_used}/{max_llm_calls})",
        }

    # 2. Resolve active LLM provider
    if llm is None:
        _, llm, _ = get_providers()

    # 3. Load and format prompt template
    categories_block, priorities_block = _format_taxonomy_blocks()
    prompt_template = load_prompt("triage")
    prompt_text = prompt_template.format(
        categories_block=categories_block,
        priorities_block=priorities_block,
        customer_name=state.get("customer_name", "Customer"),
        channel=state.get("channel", "web"),
        ticket_subject=state.get("subject", ""),
        ticket_body=state.get("body", ""),
    )

    # 4. Invoke LLM via structured output helper
    try:
        output, attempts = call_structured(
            llm=llm,
            prompt=prompt_text,
            schema=TriageOutput,
            max_retries=2,
            return_metadata=True,
            max_tokens=150,
        )
        llm_calls_used += attempts

        # 5. Domain rule: Escalate to P1 on high churn risk OR (angry + refund_billing)
        assigned_priority = output.priority.upper()
        if output.churn_risk.lower() == "high" or (
            output.sentiment.lower() == "angry" and output.category.lower() == "refund_billing"
        ):
            assigned_priority = TicketPriority.P1.value

        logger.info(
            "Triage completed successfully",
            extra={
                "ticket_id": ticket_id,
                "category": output.category,
                "priority": assigned_priority,
                "confidence": output.confidence,
                "sentiment": output.sentiment,
                "churn_risk": output.churn_risk,
                "attempts": attempts,
                "llm_calls_used": llm_calls_used,
            },
        )

        return {
            "category": output.category.lower(),
            "priority": assigned_priority,
            "sentiment": output.sentiment.lower(),
            "churn_risk": output.churn_risk.lower(),
            "confidence": output.confidence,
            "triage_reasoning": output.reasoning,
            "llm_calls_used": llm_calls_used,
            "status": TicketStatus.TRIAGING.value,
        }

    except StructuredOutputError as exc:
        llm_calls_used += exc.attempts
        logger.error(
            "Triage failed after structured retry exhaustion; routing to untriaged",
            extra={
                "ticket_id": ticket_id,
                "attempts": exc.attempts,
                "error": str(exc),
                "llm_calls_used": llm_calls_used,
            },
        )
        return {
            "status": TicketStatus.UNTRIAGED.value,
            "confidence": 0.0,
            "llm_calls_used": llm_calls_used,
            "error": f"Structured triage parsing failed: {exc}",
        }
