"""
app/db/models.py

Responsibility: SQLAlchemy 2.0 ORM models for Tickets, Drafts, and Settings.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.db.enums import TicketChannel, TicketStatus


class Base(DeclarativeBase):
    """Declarative base class for all SQLAlchemy ORM models."""
    pass


def _utcnow() -> datetime:
    """Return current UTC timestamp without timezone offset issues in SQLite."""
    return datetime.now(UTC)


class Ticket(Base):
    """
    Core support ticket entity tracking customer message, triage classification,
    resolution status, and knowledge gap flags.
    """

    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    channel: Mapped[str] = mapped_column(String(32), default=TicketChannel.WEB.value, nullable=False)
    customer_name: Mapped[str] = mapped_column(String(128), nullable=False)
    subject: Mapped[str] = mapped_column(String(256), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)

    # Lifecycle & Classification
    status: Mapped[str] = mapped_column(String(32), default=TicketStatus.NEW.value, nullable=False, index=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    priority: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    sentiment: Mapped[str | None] = mapped_column(String(32), nullable=True)
    churn_risk: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Resolution Telemetry
    kb_gap: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    critic_verdict: Mapped[str | None] = mapped_column(String(32), nullable=True)
    critic_violations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, default=list, nullable=True)
    critic_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    # Relationships
    drafts: Mapped[list[Draft]] = relationship(
        "Draft",
        back_populates="ticket",
        cascade="all, delete-orphan",
        order_by="Draft.version",
    )
    decisions: Mapped[list[ApprovalDecision]] = relationship(
        "ApprovalDecision",
        back_populates="ticket",
        cascade="all, delete-orphan",
        order_by="ApprovalDecision.decided_at",
    )
    outbox_entries: Mapped[list[Outbox]] = relationship(
        "Outbox",
        back_populates="ticket",
        cascade="all, delete-orphan",
        order_by="Outbox.sent_at",
    )


class Draft(Base):
    """
    Response drafts produced by the Resolver node for a specific ticket.
    Multiple versions may exist if revisions occur.
    """

    __tablename__ = "drafts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(String(64), ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    created_by_node: Mapped[str] = mapped_column(String(64), default="resolver", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    ticket: Mapped[Ticket] = relationship("Ticket", back_populates="drafts")


class Outbox(Base):
    """
    Mock outbox table recording sent customer communications.
    """

    __tablename__ = "outbox"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tickets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    final_body: Mapped[str] = mapped_column(Text, nullable=False)
    sent_by: Mapped[str] = mapped_column(String(32), nullable=False)  # "auto" | "human"
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    ticket: Mapped[Ticket] = relationship("Ticket", back_populates="outbox_entries")


class ApprovalDecision(Base):
    """
    Audit log of human approval actions (approve, edit, reject).
    Feeds online evaluation and calibration metrics in Phase 6.
    """

    __tablename__ = "approval_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tickets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    decision: Mapped[str] = mapped_column(String(32), nullable=False)  # "approve" | "edit" | "reject"
    original_body: Mapped[str | None] = mapped_column(Text, nullable=True)
    edited_body: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    ticket: Mapped[Ticket] = relationship("Ticket", back_populates="decisions")


class Setting(Base):
    """
    Persistent application settings, such as auto_send_enabled.
    """

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(String(256), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)
