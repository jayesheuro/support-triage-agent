"""
app/services/tracing/tracer.py

Responsibility: Core First-Party Tracer service for recording execution spans and runs.
Provides context managers and decorators for instrumenting agent graph nodes, LLM calls,
confidence/action gates, and HITL interrupt/resume lifecycles.

Key guarantees:
1. Automatic parent/child span hierarchy nesting via contextvars.
2. In-process structured logging alignment on span completion.
3. NEVER throws: any tracing or database failure is safely caught and logged.
4. Resumes continue the same execution run with explicit resume spans.
"""

from __future__ import annotations

import contextvars
import functools
import time
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, ParamSpec, TypeVar

from app.core.logging import get_logger
from app.db.enums import SpanStatus, SpanType
from app.services.tracing.store import TraceStore, get_trace_store

logger = get_logger(__name__)

# Context variables tracking active execution scope
_CURRENT_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_run_id", default=None)
_CURRENT_TICKET_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_ticket_id", default=None)
_CURRENT_PARENT_SPAN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_parent_span_id", default=None)
_CURRENT_NODE_NAME: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_node_name", default=None)
_CURRENT_DB_SESSION: contextvars.ContextVar[Any | None] = contextvars.ContextVar("current_db_session", default=None)


def get_current_run_id() -> str | None:
    return _CURRENT_RUN_ID.get()


def get_current_ticket_id() -> str | None:
    return _CURRENT_TICKET_ID.get()


def get_current_parent_span_id() -> str | None:
    return _CURRENT_PARENT_SPAN_ID.get()


def get_current_node_name() -> str | None:
    return _CURRENT_NODE_NAME.get()


def get_current_db_session() -> Any | None:
    return _CURRENT_DB_SESSION.get()


class SpanContext:
    """
    Mutable context yielded by tracer.span(). Callers update decision_summary,
    token counts, status, and payload excerpts during execution.
    """

    def __init__(
        self,
        span_id: str,
        name: str,
        span_type: str,
        parent_span_id: str | None,
        started_at: datetime,
    ) -> None:
        self.span_id = span_id
        self.name = name
        self.span_type = span_type
        self.parent_span_id = parent_span_id
        self.started_at = started_at
        self.finished_at: datetime | None = None
        self.duration_ms: float | None = None

        self.status: str = SpanStatus.OK.value
        self.retry_count: int = 0
        self.model: str | None = None
        self.tokens_in: int | None = None
        self.tokens_out: int | None = None
        self.prompt_name: str | None = None
        self.prompt_version: str | None = None
        self.policy_version: str | None = None
        self.decision_summary: str | None = None
        self.input_excerpt: str | None = None
        self.output_excerpt: str | None = None
        self.is_customer_content: bool = False
        self.attributes: dict[str, Any] = {}

    def set_tokens(self, tokens_in: int | None, tokens_out: int | None) -> None:
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out

    def set_decision(self, summary: str) -> None:
        self.decision_summary = summary

    def set_retried(self, retry_count: int = 1) -> None:
        self.status = SpanStatus.RETRIED.value
        self.retry_count = retry_count


class Tracer:
    """
    Singleton or injected Tracer coordinating trace runs and spans.
    """

    def __init__(self, store: TraceStore | None = None) -> None:
        self.store = store or get_trace_store()

    def start_run(
        self,
        ticket_id: str,
        run_id: str | None = None,
        started_at: datetime | None = None,
        db: Any | None = None,
    ) -> str:
        """
        Start or bind an active execution run for a ticket.
        Sets context variables and initializes the database TraceRun row.
        """
        active_run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        _CURRENT_RUN_ID.set(active_run_id)
        _CURRENT_TICKET_ID.set(ticket_id)
        _CURRENT_PARENT_SPAN_ID.set(None)
        _CURRENT_NODE_NAME.set(None)
        if db is not None:
            _CURRENT_DB_SESSION.set(db)

        try:
            self.store.create_or_get_run(
                ticket_id=ticket_id,
                run_id=active_run_id,
                started_at=started_at or datetime.now(UTC),
                db=_CURRENT_DB_SESSION.get(),
            )
        except Exception as exc:
            logger.error("Failed to start trace run: %s", exc)

        return active_run_id

    def finalize_run(
        self,
        final_status: str,
        run_id: str | None = None,
        db: Any | None = None,
    ) -> None:
        """Finalize the current run with its completion status."""
        active_run_id = run_id or _CURRENT_RUN_ID.get()
        if not active_run_id:
            return
        active_db = db if db is not None else _CURRENT_DB_SESSION.get()
        try:
            self.store.finalize_run(run_id=active_run_id, final_status=final_status, db=active_db)
        except Exception as exc:
            logger.error("Failed to finalize trace run: %s", exc)

    @contextmanager
    def span(
        self,
        name: str,
        span_type: str = SpanType.NODE.value,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
        policy_version: str | None = None,
        is_customer_content: bool = False,
        input_excerpt: str | None = None,
    ) -> Generator[SpanContext, None, None]:
        """
        Context manager to record a trace span. Automatically manages span hierarchy
        and logs span completion. Never throws on tracing persistence failures.
        """
        run_id = _CURRENT_RUN_ID.get()
        ticket_id = _CURRENT_TICKET_ID.get()

        # If no active run, create a transient fallback run
        if not run_id:
            run_id = f"run-{uuid.uuid4().hex[:12]}"
            _CURRENT_RUN_ID.set(run_id)
            if not ticket_id:
                ticket_id = "unknown"
                _CURRENT_TICKET_ID.set(ticket_id)
            self.store.create_or_get_run(ticket_id=ticket_id, run_id=run_id)

        span_id = f"span-{uuid.uuid4().hex[:10]}"
        parent_span_id = _CURRENT_PARENT_SPAN_ID.get()
        started_at = datetime.now(UTC)

        ctx = SpanContext(
            span_id=span_id,
            name=name,
            span_type=span_type,
            parent_span_id=parent_span_id,
            started_at=started_at,
        )
        ctx.prompt_name = prompt_name
        ctx.prompt_version = prompt_version
        ctx.policy_version = policy_version
        ctx.is_customer_content = is_customer_content
        ctx.input_excerpt = input_excerpt

        # Set active context for child spans and logging
        token_parent = _CURRENT_PARENT_SPAN_ID.set(span_id)
        token_node = _CURRENT_NODE_NAME.set(name) if span_type == SpanType.NODE.value else None

        t0 = time.monotonic()
        try:
            yield ctx
        except Exception as exc:
            ctx.status = SpanStatus.FAILED.value
            ctx.attributes["error"] = str(exc)
            raise
        finally:
            finished_at = datetime.now(UTC)
            duration_ms = max(0.0, (time.monotonic() - t0) * 1000)
            ctx.finished_at = finished_at
            ctx.duration_ms = duration_ms

            # Reset context variables
            _CURRENT_PARENT_SPAN_ID.reset(token_parent)
            if token_node is not None:
                _CURRENT_NODE_NAME.reset(token_node)

            # Persist span safely
            try:
                self.store.record_span(
                    run_id=run_id,
                    span_id=ctx.span_id,
                    name=ctx.name,
                    span_type=ctx.span_type,
                    parent_span_id=ctx.parent_span_id,
                    started_at=ctx.started_at,
                    finished_at=ctx.finished_at,
                    duration_ms=ctx.duration_ms,
                    status=ctx.status,
                    retry_count=ctx.retry_count,
                    model=ctx.model,
                    tokens_in=ctx.tokens_in,
                    tokens_out=ctx.tokens_out,
                    prompt_name=ctx.prompt_name,
                    prompt_version=ctx.prompt_version,
                    policy_version=ctx.policy_version,
                    decision_summary=ctx.decision_summary,
                    input_excerpt=ctx.input_excerpt,
                    output_excerpt=ctx.output_excerpt,
                    is_customer_content=ctx.is_customer_content,
                    attributes=ctx.attributes,
                    db=_CURRENT_DB_SESSION.get(),
                )
            except Exception as store_exc:
                logger.error("Tracer store.record_span failed: %s", store_exc)

            # Structured logging alignment: 1 INFO line per span completion
            logger.info(
                "Span '%s' completed [%s] (%.1f ms): %s",
                ctx.name,
                ctx.status,
                ctx.duration_ms,
                ctx.decision_summary or "completed",
                extra={
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "node": ctx.name,
                    "span_id": ctx.span_id,
                    "span_type": ctx.span_type,
                    "duration_ms": round(ctx.duration_ms, 1),
                    "status": ctx.status,
                    "decision_summary": ctx.decision_summary,
                },
            )

    def record_event(
        self,
        name: str,
        span_type: str,
        decision_summary: str,
        status: str = SpanStatus.OK.value,
        attributes: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
    ) -> SpanContext | None:
        """
        Record an instantaneous or zero-duration lifecycle event (e.g. gates, interrupts, resumes).
        """
        run_id = _CURRENT_RUN_ID.get()
        ticket_id = _CURRENT_TICKET_ID.get()
        if not run_id:
            run_id = f"run-{uuid.uuid4().hex[:12]}"
            _CURRENT_RUN_ID.set(run_id)
            if not ticket_id:
                ticket_id = "unknown"
                _CURRENT_TICKET_ID.set(ticket_id)
            self.store.create_or_get_run(ticket_id=ticket_id, run_id=run_id)

        span_id = f"span-{uuid.uuid4().hex[:10]}"
        parent_span_id = _CURRENT_PARENT_SPAN_ID.get()
        now = datetime.now(UTC)

        try:
            self.store.record_span(
                run_id=run_id,
                span_id=span_id,
                name=name,
                span_type=span_type,
                parent_span_id=parent_span_id,
                started_at=now,
                finished_at=now,
                duration_ms=duration_ms,
                status=status,
                decision_summary=decision_summary,
                attributes=attributes or {},
                db=_CURRENT_DB_SESSION.get(),
            )
            logger.info(
                "Event span '%s' [%s]: %s",
                name,
                span_type,
                decision_summary,
                extra={
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "node": name,
                    "span_id": span_id,
                    "span_type": span_type,
                    "duration_ms": duration_ms,
                    "status": status,
                    "decision_summary": decision_summary,
                },
            )
        except Exception as exc:
            logger.error("Tracer record_event failed: %s", exc)
            return None

        ctx = SpanContext(span_id, name, span_type, parent_span_id, now)
        ctx.decision_summary = decision_summary
        ctx.status = status
        return ctx


_DEFAULT_TRACER: Tracer | None = None


def get_tracer() -> Tracer:
    """Retrieve global singleton Tracer instance."""
    global _DEFAULT_TRACER
    if _DEFAULT_TRACER is None:
        _DEFAULT_TRACER = Tracer()
    return _DEFAULT_TRACER


def traced_node(name: str | None = None):
    """
    Decorator for wrapping an agent node function in a trace span.
    Extracts ticket_id and run_id from AgentState if not already set.
    """
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        node_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = get_tracer()
            # If state dictionary is passed as first arg, extract context
            if args and isinstance(args[0], dict):
                state_dict = args[0]
                ticket_id = state_dict.get("ticket_id")
                run_id = state_dict.get("run_id")
                if ticket_id and not _CURRENT_TICKET_ID.get():
                    _CURRENT_TICKET_ID.set(ticket_id)
                if run_id and not _CURRENT_RUN_ID.get():
                    tracer.start_run(ticket_id=ticket_id or "unknown", run_id=run_id)

            with tracer.span(name=node_name, span_type=SpanType.NODE.value) as span_ctx:
                result = fn(*args, **kwargs)
                if isinstance(result, dict):
                    # Auto-extract decision summary if not manually set
                    if not span_ctx.decision_summary:
                        if "category" in result:
                            conf = result.get("confidence", 0.0)
                            prio = result.get("priority", "")
                            cat = result.get("category", "")
                            span_ctx.set_decision(f"category={cat} prio={prio} conf={conf:.2f}")
                        elif "critic_verdict" in result:
                            verdict = result.get("critic_verdict")
                            viols = len(result.get("critic_violations") or [])
                            span_ctx.set_decision(f"verdict={verdict} violations={viols}")
                        elif "draft_body" in result:
                            citations = result.get("citations", [])
                            span_ctx.set_decision(f"drafted citations={citations}")
                return result

        return wrapper
    return decorator
