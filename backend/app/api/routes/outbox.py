"""
app/api/routes/outbox.py

Responsibility: REST API route for inspecting sent communications in the mock outbox.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.db.models import Outbox
from app.models.outbox import OutboxEntryResponse

router = APIRouter(tags=["outbox"])


@router.get("/outbox", response_model=list[OutboxEntryResponse])
def list_outbox_entries(db: Session = Depends(get_db)) -> list[OutboxEntryResponse]:
    """
    List all mock communications sent via auto-send or human approval.
    """
    entries = db.query(Outbox).order_by(Outbox.sent_at.desc()).all()
    return [
        OutboxEntryResponse(
            id=entry.id,
            ticket_id=entry.ticket_id,
            final_body=entry.final_body,
            sent_by=entry.sent_by,
            sent_at=entry.sent_at,
        )
        for entry in entries
    ]
