"""
app/models/approvals.py

Responsibility: Pydantic request and response schemas for human approval queue endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ApprovalDecisionRequest(BaseModel):
    """Payload submitted by a human reviewer to approve, edit, or reject a pending ticket."""

    action: Literal["approve", "edit", "reject"] = Field(
        ...,
        description="Reviewer decision action",
    )
    edited_body: str | None = Field(
        default=None,
        description="Revised customer response text (mandatory when action is 'edit')",
    )
    reason: str | None = Field(
        default=None,
        description="Justification for rejection (mandatory when action is 'reject')",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> ApprovalDecisionRequest:
        if self.action == "edit" and not (self.edited_body and self.edited_body.strip()):
            raise ValueError("edited_body is required when action is 'edit'")
        if self.action == "reject" and not (self.reason and self.reason.strip()):
            raise ValueError("reason is required when action is 'reject'")
        return self


class PendingApprovalTicket(BaseModel):
    """Pending ticket metadata and draft for human reviewer inspection."""

    ticket_id: str
    customer_name: str
    channel: str
    subject: str
    body: str
    category: str | None
    priority: str | None
    sentiment: str | None
    churn_risk: str | None
    confidence: float | None
    kb_gap: bool
    draft_version: int | None
    draft_body: str | None
    citations: list[str] = Field(default_factory=list)
    critic_verdict: str | None
    critic_violations: list[dict[str, Any]] = Field(default_factory=list)
    critic_reasoning: str | None
    created_at: datetime


class ApprovalDecisionResponse(BaseModel):
    """Response returned upon successfully recording a human decision."""

    ticket_id: str
    status: str
    action: str
    final_body: str | None
    decided_at: datetime
