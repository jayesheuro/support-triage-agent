"""
app/api/routes/traces.py

Responsibility: REST API endpoints for agent observability and execution tracing.
Exposes per-ticket ordered trace history (/api/tickets/{id}/trace) and system-wide
latency, retry, verdict, and cost metrics (/api/traces/stats).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, ValidationError
from app.db import get_db
from app.db.models import Ticket
from app.models.traces import TicketTraceResponse, TraceStatsResponse
from app.services.tracing.store import get_trace_store

router = APIRouter(tags=["traces"])


@router.get("/tickets/{ticket_id}/trace", response_model=TicketTraceResponse, summary="Get execution trace for a ticket")
def get_ticket_trace(
    ticket_id: str,
    db: Session = Depends(get_db),
) -> TicketTraceResponse:
    """
    Retrieve full execution trace runs and ordered spans for a specific ticket.
    404-safe: if ticket exists but has no trace runs (e.g. processed before tracing),
    returns an empty runs list.
    """
    ticket = db.query(Ticket).filter_by(id=ticket_id).first()
    if not ticket:
        raise NotFoundError(f"Ticket '{ticket_id}' not found")

    store = get_trace_store()
    raw_runs = store.get_ticket_trace(ticket_id=ticket_id, db=db)

    return TicketTraceResponse(ticket_id=ticket.id, runs=raw_runs)


@router.get("/traces/stats", response_model=TraceStatsResponse, summary="Get trace and cost observability stats")
def get_trace_stats(
    window: Annotated[str, Query(description="Aggregation window: 1h, 24h, 7d, all")] = "24h",
    db: Session = Depends(get_db),
) -> TraceStatsResponse:
    """
    Aggregate observability metrics across trace runs and spans:
    - Per-node average and P95 latency
    - Token consumption and estimated dollar cost on Gemini 2.0 Flash
    - Retry rates per node
    - Knowledge gap rate
    - Critic verdict distribution (pass, revise, escalate)
    - Auto-send vs human-review ratio
    """
    valid_windows = {"1h", "24h", "7d", "all"}
    if window not in valid_windows:
        raise ValidationError(f"Invalid window '{window}'. Allowed values: {sorted(list(valid_windows))}")

    store = get_trace_store()
    stats = store.get_trace_stats(window=window, db=db)

    return TraceStatsResponse(**stats)
