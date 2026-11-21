"""
app/db/models.py

Responsibility: SQLAlchemy 2.0 ORM models for Tickets, Drafts, Settings,
Outbox, ApprovalDecisions, and Trace telemetry (TraceRun, TraceSpan).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.db.enums import SpanStatus, SpanType, TicketChannel, TicketStatus


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
    trace_runs: Mapped[list[TraceRun]] = relationship(
        "TraceRun",
        back_populates="ticket",
        cascade="all, delete-orphan",
        order_by="TraceRun.started_at",
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


class TraceRun(Base):
    """
    Top-level execution trace for a single ticket graph run.
    Contains overall execution metrics and links to detailed child spans.
    """

    __tablename__ = "trace_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    ticket_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tickets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_llm_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens_in: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens_out: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    final_status: Mapped[str] = mapped_column(String(32), default=TicketStatus.TRIAGING.value, nullable=False)

    ticket: Mapped[Ticket] = relationship("Ticket", back_populates="trace_runs")
    spans: Mapped[list[TraceSpan]] = relationship(
        "TraceSpan",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="TraceSpan.started_at",
    )


class TraceSpan(Base):
    """
    Granular event span representing a node execution, LLM invocation, gate evaluation,
    or interrupt/resume lifecycle milestone.
    """

    __tablename__ = "trace_spans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("trace_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    span_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    parent_span_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    span_type: Mapped[str] = mapped_column(String(32), default=SpanType.NODE.value, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default=SpanStatus.OK.value, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    policy_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    decision_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_customer_content: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    attributes: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict, nullable=True)

    run: Mapped[TraceRun] = relationship("TraceRun", back_populates="spans")
