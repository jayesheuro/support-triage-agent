"""
app/api/routes/approvals.py

Responsibility: REST API routes for human-in-the-loop approval queue.
Enables human agents to review pending tickets, inspect drafts and critic verdicts,
and submit approve/edit/reject decisions that resume the LangGraph execution thread.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.agents.graph import resume_ticket_graph
from app.db import get_db
from app.db.enums import TicketStatus
from app.db.models import Draft, Ticket
from app.models.approvals import (
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    PendingApprovalTicket,
)

router = APIRouter(tags=["approvals"])


@router.get("/approvals", response_model=list[PendingApprovalTicket])
def list_pending_approvals(db: Session = Depends(get_db)) -> list[PendingApprovalTicket]:
    """
    List all support tickets currently awaiting human approval.
    Includes draft contents, classification telemetry, and critic compliance verdicts.
    """
    tickets = (
        db.query(Ticket)
        .filter(Ticket.status == TicketStatus.PENDING_REVIEW.value)
        .order_by(Ticket.created_at.asc())
        .all()
    )

    results: list[PendingApprovalTicket] = []
    for t in tickets:
        latest_draft = (
            db.query(Draft)
            .filter_by(ticket_id=t.id)
            .order_by(Draft.version.desc())
            .first()
        )

        results.append(
            PendingApprovalTicket(
                ticket_id=t.id,
                customer_name=t.customer_name,
                channel=t.channel,
                subject=t.subject,
                body=t.body,
                category=t.category,
                priority=t.priority,
                sentiment=t.sentiment,
                churn_risk=t.churn_risk,
                confidence=t.confidence,
                kb_gap=t.kb_gap,
                draft_version=latest_draft.version if latest_draft else None,
                draft_body=latest_draft.body if latest_draft else None,
                citations=latest_draft.citations if latest_draft else [],
                critic_verdict=t.critic_verdict,
                critic_violations=t.critic_violations or [],
                critic_reasoning=t.critic_reasoning,
                created_at=t.created_at,
            )
        )

    return results


@router.post("/approvals/{ticket_id}/decision", response_model=ApprovalDecisionResponse)
def submit_approval_decision(
    ticket_id: str,
    decision_req: ApprovalDecisionRequest,
    db: Session = Depends(get_db),
) -> ApprovalDecisionResponse:
    """
    Submit a human reviewer decision (approve, edit, reject) on a pending ticket.
    Resumes the LangGraph thread from its checkpoint, records decision metrics,
    and dispatches accepted replies to the mock outbox.
    """
    decision_dict = decision_req.model_dump()
    resume_ticket_graph(ticket_id=ticket_id, decision=decision_dict, db=db)

    # Reload updated ticket
    ticket = db.query(Ticket).filter_by(id=ticket_id).first()
    latest_draft = (
        db.query(Draft)
        .filter_by(ticket_id=ticket_id)
        .order_by(Draft.version.desc())
        .first()
    )

    final_body = (
        decision_req.edited_body
        if decision_req.action == "edit"
        else (latest_draft.body if latest_draft and decision_req.action == "approve" else None)
    )

    return ApprovalDecisionResponse(
        ticket_id=ticket_id,
        status=ticket.status if ticket else "unknown",
        action=decision_req.action,
        final_body=final_body,
        decided_at=datetime.now(UTC),
    )
