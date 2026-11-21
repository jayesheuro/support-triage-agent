"""
app/services/tracing/store.py

Responsibility: Persistence and retrieval of agent execution traces (TraceRun, TraceSpan)
in SQLite. Provides query methods for ticket trace retrieval and dashboard statistical aggregations.
Crucial invariant: NEVER throws to callers — errors are caught and logged so agent workflows
are never interrupted by tracing failures.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.enums import SpanStatus, SpanType, TicketStatus
from app.db.models import Ticket, TraceRun, TraceSpan
from app.db.session import SessionLocal

logger = get_logger(__name__)

DEFAULT_TRUNCATION_LIMIT = 500


def truncate_payload(text: str | None, max_len: int = DEFAULT_TRUNCATION_LIMIT) -> str | None:
    """
    Truncate payload strings to ~500 chars to respect the privacy default.
    Appends '... [truncated]' if string exceeds max_len.
    """
    if text is None:
        return None
    if len(text) <= max_len:
        return text
    return text[:max_len] + "... [truncated]"


class TraceStore:
    """
    SQLite-backed store for TraceRun and TraceSpan records.
    Designed to be robust and fail-safe: every database write operation
    is wrapped in try/except.
    """

    def __init__(self, session_factory=SessionLocal) -> None:
        self.session_factory = session_factory

    def create_or_get_run(
        self,
        ticket_id: str,
        run_id: str,
        started_at: datetime | None = None,
        db: Session | None = None,
    ) -> TraceRun | None:
        """
        Create a new TraceRun record or return the existing one for run_id.
        Fails safe and returns None on error.
        """
        def _execute(session: Session) -> TraceRun:
            run = session.query(TraceRun).filter_by(run_id=run_id).first()
            if not run:
                run = TraceRun(
                    run_id=run_id,
                    ticket_id=ticket_id,
                    started_at=started_at or datetime.now(UTC),
                    final_status=TicketStatus.TRIAGING.value,
                )
                session.add(run)
                session.commit()
                session.refresh(run)
            return run

        try:
            if db is not None:
                return _execute(db)
            with self.session_factory() as session:
                return _execute(session)
        except Exception as exc:
            logger.error("TraceStore error in create_or_get_run: %s", exc, extra={"run_id": run_id, "ticket_id": ticket_id})
            return None

    def record_span(
        self,
        run_id: str,
        span_id: str,
        name: str,
        span_type: str = SpanType.NODE.value,
        parent_span_id: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        duration_ms: float | None = None,
        status: str = SpanStatus.OK.value,
        retry_count: int = 0,
        model: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
        policy_version: str | None = None,
        decision_summary: str | None = None,
        input_excerpt: str | None = None,
        output_excerpt: str | None = None,
        is_customer_content: bool = False,
        attributes: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> TraceSpan | None:
        """
        Insert a completed TraceSpan record and update aggregate token/call counts on the run.
        Truncates input and output payloads according to the privacy default.
        """
        trunc_in = truncate_payload(input_excerpt)
        trunc_out = truncate_payload(output_excerpt)

        def _execute(session: Session) -> TraceSpan:
            span = TraceSpan(
                run_id=run_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                span_type=span_type,
                name=name,
                started_at=started_at or datetime.now(UTC),
                finished_at=finished_at,
                duration_ms=duration_ms,
                status=status,
                retry_count=retry_count,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                policy_version=policy_version,
                decision_summary=decision_summary,
                input_excerpt=trunc_in,
                output_excerpt=trunc_out,
                is_customer_content=is_customer_content,
                attributes=attributes or {},
            )
            session.add(span)

            # Update parent run metrics if llm_call
            run = session.query(TraceRun).filter_by(run_id=run_id).first()
            if run:
                if span_type == SpanType.LLM_CALL.value or model is not None:
                    run.total_llm_calls += 1
                if tokens_in:
                    run.total_tokens_in += tokens_in
                if tokens_out:
                    run.total_tokens_out += tokens_out

            session.commit()
            session.refresh(span)
            return span

        try:
            if db is not None:
                return _execute(db)
            with self.session_factory() as session:
                return _execute(session)
        except Exception as exc:
            logger.error("TraceStore error in record_span: %s", exc, extra={"run_id": run_id, "span_id": span_id, "name": name})
            return None

    def finalize_run(
        self,
        run_id: str,
        final_status: str,
        finished_at: datetime | None = None,
        db: Session | None = None,
    ) -> TraceRun | None:
        """
        Update the final status, finished timestamp, and total duration of a TraceRun.
        """
        def _execute(session: Session) -> TraceRun | None:
            run = session.query(TraceRun).filter_by(run_id=run_id).first()
            if not run:
                return None
            end_time = finished_at or datetime.now(UTC)
            run.finished_at = end_time
            run.final_status = final_status
            if run.started_at:
                st = run.started_at
                if st.tzinfo is None:
                    st = st.replace(tzinfo=UTC)
                et = end_time
                if et.tzinfo is None:
                    et = et.replace(tzinfo=UTC)
                run.total_duration_ms = max(0.0, (et - st).total_seconds() * 1000)
            session.commit()
            session.refresh(run)
            return run

        try:
            if db is not None:
                return _execute(db)
            with self.session_factory() as session:
                return _execute(session)
        except Exception as exc:
            logger.error("TraceStore error in finalize_run: %s", exc, extra={"run_id": run_id})
            return None

    def get_ticket_trace(self, ticket_id: str, db: Session | None = None) -> list[dict[str, Any]]:
        """
        Retrieve all execution runs and their ordered spans for a ticket.
        Returns a serializable list of run objects with nested spans.
        """
        def _execute(session: Session) -> list[dict[str, Any]]:
            runs = (
                session.query(TraceRun)
                .filter_by(ticket_id=ticket_id)
                .order_by(TraceRun.started_at.asc())
                .all()
            )
            results: list[dict[str, Any]] = []
            for r in runs:
                spans = (
                    session.query(TraceSpan)
                    .filter_by(run_id=r.run_id)
                    .order_by(TraceSpan.started_at.asc(), TraceSpan.id.asc())
                    .all()
                )
                results.append(
                    {
                        "run_id": r.run_id,
                        "ticket_id": r.ticket_id,
                        "started_at": r.started_at.isoformat() if r.started_at else None,
                        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                        "total_duration_ms": r.total_duration_ms,
                        "total_llm_calls": r.total_llm_calls,
                        "total_tokens_in": r.total_tokens_in,
                        "total_tokens_out": r.total_tokens_out,
                        "final_status": r.final_status,
                        "spans": [
                            {
                                "id": s.id,
                                "span_id": s.span_id,
                                "parent_span_id": s.parent_span_id,
                                "span_type": s.span_type,
                                "name": s.name,
                                "started_at": s.started_at.isoformat() if s.started_at else None,
                                "finished_at": s.finished_at.isoformat() if s.finished_at else None,
                                "duration_ms": s.duration_ms,
                                "status": s.status,
                                "retry_count": s.retry_count,
                                "model": s.model,
                                "tokens_in": s.tokens_in,
                                "tokens_out": s.tokens_out,
                                "prompt_name": s.prompt_name,
                                "prompt_version": s.prompt_version,
                                "policy_version": s.policy_version,
                                "decision_summary": s.decision_summary,
                                "input_excerpt": s.input_excerpt,
                                "output_excerpt": s.output_excerpt,
                                "is_customer_content": s.is_customer_content,
                                "attributes": s.attributes or {},
                            }
                            for s in spans
                        ],
                    }
                )
            return results

        try:
            if db is not None:
                return _execute(db)
            with self.session_factory() as session:
                return _execute(session)
        except Exception as exc:
            logger.error("TraceStore error in get_ticket_trace: %s", exc, extra={"ticket_id": ticket_id})
            return []

    def get_trace_stats(
        self,
        window: str = "24h",
        db: Session | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate observability statistics across trace runs and spans.
        Supports window filters: 1h, 24h, 7d, all.
        """
        def _execute(session: Session) -> dict[str, Any]:
            now = datetime.now(UTC)
            start_time: datetime | None = None
            if window == "1h":
                start_time = now - timedelta(hours=1)
            elif window == "24h":
                start_time = now - timedelta(hours=24)
            elif window == "7d":
                start_time = now - timedelta(days=7)
            # "all" leaves start_time as None

            run_query = session.query(TraceRun)
            span_query = session.query(TraceSpan)

            if start_time:
                run_query = run_query.filter(TraceRun.started_at >= start_time)
                span_query = span_query.filter(TraceSpan.started_at >= start_time)

            runs = run_query.all()
            spans = span_query.all()

            total_runs = len(runs)
            unique_tickets = len({r.ticket_id for r in runs})
            total_spans = len(spans)

            # Token and call totals
            total_llm_calls = sum(r.total_llm_calls for r in runs)
            total_prompt_tokens = sum(r.total_tokens_in for r in runs)
            total_completion_tokens = sum(r.total_tokens_out for r in runs)
            total_tokens = total_prompt_tokens + total_completion_tokens

            avg_tokens_per_ticket = (
                round(total_tokens / unique_tickets, 1) if unique_tickets > 0 else 0.0
            )

            # Cost estimation via Gemini pricing config
            settings = get_settings()
            in_rate = getattr(settings.app, "gemini_cost_per_million_input_tokens", 0.10)
            out_rate = getattr(settings.app, "gemini_cost_per_million_output_tokens", 0.40)
            estimated_cost_usd = round(
                (total_prompt_tokens / 1_000_000 * in_rate)
                + (total_completion_tokens / 1_000_000 * out_rate),
                6,
            )

            # Node latency aggregation (avg & p95)
            node_durations_raw: dict[str, list[float]] = {}
            node_retry_counts: dict[str, int] = {}
            node_total_counts: dict[str, int] = {}

            critic_verdicts: dict[str, int] = {"pass": 0, "revise": 0, "escalate": 0}
            resolver_spans = 0
            kb_gap_count = 0

            for s in spans:
                if s.span_type == SpanType.NODE.value:
                    node_name = s.name
                    node_total_counts[node_name] = node_total_counts.get(node_name, 0) + 1
                    if s.duration_ms is not None:
                        node_durations_raw.setdefault(node_name, []).append(s.duration_ms)
                    if s.status == SpanStatus.RETRIED.value or s.retry_count > 0:
                        node_retry_counts[node_name] = node_retry_counts.get(node_name, 0) + 1

                # Critic verdict metrics
                if s.name == "critic" and s.decision_summary:
                    lower_summary = s.decision_summary.lower()
                    if "verdict=pass" in lower_summary:
                        critic_verdicts["pass"] += 1
                    elif "verdict=revise" in lower_summary:
                        critic_verdicts["revise"] += 1
                    elif "verdict=escalate" in lower_summary:
                        critic_verdicts["escalate"] += 1

                # KB gap metrics
                if s.name == "resolver":
                    resolver_spans += 1
                    if s.decision_summary and "kb_gap=true" in s.decision_summary.lower():
                        kb_gap_count += 1

            node_durations: dict[str, dict[str, float | int]] = {}
            for name, durations in node_durations_raw.items():
                if not durations:
                    continue
                sorted_d = sorted(durations)
                avg_d = round(sum(sorted_d) / len(sorted_d), 1)
                idx_95 = math.ceil(0.95 * len(sorted_d)) - 1
                p95_d = round(sorted_d[max(0, min(idx_95, len(sorted_d) - 1))], 1)
                node_durations[name] = {
                    "avg_ms": avg_d,
                    "p95_ms": p95_d,
                    "count": len(sorted_d),
                }

            # Retry rates per node
            retry_rates: dict[str, float] = {}
            for name, total_cnt in node_total_counts.items():
                retries = node_retry_counts.get(name, 0)
                retry_rates[name] = round(retries / total_cnt, 3) if total_cnt > 0 else 0.0

            kb_gap_rate = (
                round(kb_gap_count / resolver_spans, 3) if resolver_spans > 0 else 0.0
            )

            # Auto-send vs human review distribution
            auto_sent_count = sum(1 for r in runs if r.final_status == TicketStatus.AUTO_SENT.value)
            human_sent_count = sum(1 for r in runs if r.final_status == TicketStatus.SENT.value)
            pending_review_count = sum(1 for r in runs if r.final_status == TicketStatus.PENDING_REVIEW.value)
            rejected_count = sum(1 for r in runs if r.final_status == TicketStatus.REJECTED.value)

            total_decided = auto_sent_count + human_sent_count
            auto_send_ratio = (
                round(auto_sent_count / total_decided, 3) if total_decided > 0 else 0.0
            )

            return {
                "window": window,
                "total_tickets_traced": unique_tickets,
                "total_runs": total_runs,
                "total_spans": total_spans,
                "total_llm_calls": total_llm_calls,
                "tokens": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                },
                "estimated_cost_usd": estimated_cost_usd,
                "pricing_basis": {
                    "provider": "gemini-2.0-flash",
                    "input_rate_per_m": in_rate,
                    "output_rate_per_m": out_rate,
                },
                "avg_tokens_per_ticket": avg_tokens_per_ticket,
                "node_durations": node_durations,
                "retry_rates": retry_rates,
                "kb_gap_rate": kb_gap_rate,
                "verdict_distribution": critic_verdicts,
                "dispatch_stats": {
                    "auto_sent": auto_sent_count,
                    "human_sent": human_sent_count,
                    "pending_review": pending_review_count,
                    "rejected": rejected_count,
                    "auto_send_ratio": auto_send_ratio,
                },
            }

        try:
            if db is not None:
                return _execute(db)
            with self.session_factory() as session:
                return _execute(session)
        except Exception as exc:
            logger.error("TraceStore error in get_trace_stats: %s", exc)
            return {
                "window": window,
                "total_tickets_traced": 0,
                "total_runs": 0,
                "total_spans": 0,
                "total_llm_calls": 0,
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total": 0},
                "estimated_cost_usd": 0.0,
                "avg_tokens_per_ticket": 0.0,
                "node_durations": {},
                "retry_rates": {},
                "kb_gap_rate": 0.0,
                "verdict_distribution": {"pass": 0, "revise": 0, "escalate": 0},
                "dispatch_stats": {
                    "auto_sent": 0,
                    "human_sent": 0,
                    "pending_review": 0,
                    "rejected": 0,
                    "auto_send_ratio": 0.0,
                },
            }


_DEFAULT_TRACE_STORE: TraceStore | None = None


def get_trace_store() -> TraceStore:
    """Retrieve singleton TraceStore instance."""
    global _DEFAULT_TRACE_STORE
    if _DEFAULT_TRACE_STORE is None:
        _DEFAULT_TRACE_STORE = TraceStore()
    return _DEFAULT_TRACE_STORE
