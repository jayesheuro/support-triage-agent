"""
app/models/outbox.py

Responsibility: Pydantic schemas for the mock customer communications outbox.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class OutboxEntryResponse(BaseModel):
    """Schema for a sent message record in the mock outbox."""

    id: int
    ticket_id: str
    final_body: str
    sent_by: str
    sent_at: datetime
