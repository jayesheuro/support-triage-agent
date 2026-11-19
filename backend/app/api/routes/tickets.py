"""
app/api/routes/tickets.py

Responsibility: REST API endpoints for customer support ticket ingestion, retrieval,
and background graph orchestration.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy.orm import Session

from app.agents.graph import run_ticket_graph
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.enums import TicketStatus
from app.db.models import Ticket
from app.db.session import SessionLocal, get_db
from app.models.tickets import (
    DraftResponse,
    TicketCreateRequest,
    TicketCreateResponse,
    TicketDetailResponse,
    TicketListResponse,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/tickets", tags=["tickets"])


def _process_ticket_background(ticket_id: str) -> None:
    """
    Background worker function executed via FastAPI BackgroundTasks.
    Runs the agent graph synchronously in its own database session.
    """
    with SessionLocal() as db:
        try:
            logger.info("Background task starting for ticket", extra={"ticket_id": ticket_id})
            run_ticket_graph(ticket_id=ticket_id, db=db)
        except Exception as exc:
            logger.exception("Error executing background ticket graph", extra={"ticket_id": ticket_id, "error": str(exc)})
            ticket = db.query(Ticket).filter_by(id=ticket_id).first()
            if ticket:
                ticket.status = TicketStatus.UNTRIAGED.value
                db.commit()


def _format_ticket_detail(ticket: Ticket) -> TicketDetailResponse:
    """Convert SQLAlchemy Ticket model to Pydantic TicketDetailResponse."""
    return TicketDetailResponse(
        id=ticket.id,
        channel=ticket.channel,
        customer_name=ticket.customer_name,
        subject=ticket.subject,
        body=ticket.body,
        status=ticket.status,
        category=ticket.category,
        priority=ticket.priority,
        sentiment=ticket.sentiment,
        churn_risk=ticket.churn_risk,
        confidence=ticket.confidence,
        kb_gap=ticket.kb_gap,
        created_at=ticket.created_at.isoformat(),
        updated_at=ticket.updated_at.isoformat(),
        drafts=[
            DraftResponse(
                id=d.id,
                version=d.version,
                body=d.body,
                citations=d.citations or [],
                created_by_node=d.created_by_node,
                created_at=d.created_at.isoformat(),
            )
            for d in (ticket.drafts or [])
        ],
    )


@router.post(
    "",
    response_model=TicketCreateResponse,
    status_code=202,
    summary="Submit a new support ticket",
)
def create_ticket(
    payload: TicketCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TicketCreateResponse:
    """
    Accept an incoming support ticket, persist it in the database with status 'new',
    and dispatch agent graph execution to FastAPI BackgroundTasks.
    """
    ticket_id = f"TICK-{uuid.uuid4().hex[:8].upper()}"

    ticket = Ticket(
        id=ticket_id,
        channel=payload.channel,
        customer_name=payload.customer_name,
        subject=payload.subject,
        body=payload.body,
        status=TicketStatus.NEW.value,
        kb_gap=False,
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    logger.info("New ticket created; enqueuing background graph execution", extra={"ticket_id": ticket_id})
    background_tasks.add_task(_process_ticket_background, ticket_id)

    return TicketCreateResponse(
        id=ticket.id,
        status=ticket.status,
        message="Ticket created and queued for agent triage and resolution",
    )


@router.get(
    "",
    response_model=TicketListResponse,
    summary="List support tickets with optional filters",
)
def list_tickets(
    status: Annotated[str | None, Query(description="Filter by ticket status")] = None,
    category: Annotated[str | None, Query(description="Filter by category")] = None,
    priority: Annotated[str | None, Query(description="Filter by priority (P1, P2, P3)")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    db: Session = Depends(get_db),
) -> TicketListResponse:
    """
    Retrieve support tickets, ordered newest first, with optional status/category/priority filtering.
    """
    query = db.query(Ticket)

    if status:
        query = query.filter(Ticket.status == status)
    if category:
        query = query.filter(Ticket.category == category)
    if priority:
        query = query.filter(Ticket.priority == priority)

    total = query.count()
    tickets = query.order_by(Ticket.created_at.desc()).offset(offset).limit(limit).all()

    return TicketListResponse(
        tickets=[_format_ticket_detail(t) for t in tickets],
        total=total,
    )


@router.get(
    "/{ticket_id}",
    response_model=TicketDetailResponse,
    summary="Get complete details for a ticket",
)
def get_ticket(
    ticket_id: str,
    db: Session = Depends(get_db),
) -> TicketDetailResponse:
    """
    Retrieve full details for a single ticket, including its triage results, drafts, and citations.
    """
    ticket = db.query(Ticket).filter_by(id=ticket_id).first()
    if not ticket:
        raise NotFoundError(f"Ticket '{ticket_id}' not found")

    return _format_ticket_detail(ticket)
