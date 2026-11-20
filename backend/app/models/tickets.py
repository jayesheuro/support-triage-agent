"""
app/models/tickets.py

Responsibility: Pydantic schemas for the Ticket API contract.
Used by FastAPI for request validation and OpenAPI schema generation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TicketCreateRequest(BaseModel):
    """Payload for submitting a new support ticket."""

    channel: str = Field(default="web", description="Ingress channel (web, email, chat, api)")
    customer_name: str = Field(..., description="Customer's full name")
    subject: str = Field(..., description="Ticket subject line")
    body: str = Field(..., description="Customer inquiry or message text")


class TicketCreateResponse(BaseModel):
    """Response returned upon ticket creation."""

    id: str = Field(..., description="Unique ticket identifier")
    status: str = Field(..., description="Current lifecycle status")
    message: str = Field(..., description="Informational message")


class DraftResponse(BaseModel):
    """Drafted reply associated with a ticket."""

    id: int
    version: int
    body: str
    citations: list[str] = Field(default_factory=list)
    created_by_node: str
    created_at: str


class TicketDetailResponse(BaseModel):
    """Complete ticket detail including triage classifications and resolution drafts."""

    id: str
    channel: str
    customer_name: str
    subject: str
    body: str
    status: str
    category: str | None = None
    priority: str | None = None
    sentiment: str | None = None
    churn_risk: str | None = None
    confidence: float | None = None
    kb_gap: bool = False
    created_at: str
    updated_at: str
    drafts: list[DraftResponse] = Field(default_factory=list)


class TicketListResponse(BaseModel):
    """Paginated or filtered list of support tickets."""

    tickets: list[TicketDetailResponse]
    total: int
