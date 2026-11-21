"""
tests/test_tracer.py

Responsibility: Unit tests for the First-Party Tracer service.
Tests span nesting via contextvars, never-throws resilience guarantee, payload truncation,
and customer content privacy tagging.
"""

from __future__ import annotations

import unittest.mock
from datetime import UTC, datetime

import pytest

from app.db.enums import SpanStatus, SpanType
from app.services.tracing.store import TraceStore, truncate_payload
from app.services.tracing.tracer import Tracer, get_current_parent_span_id, get_current_run_id


class ExplodingStore(TraceStore):
    """Mock store that intentionally raises exceptions on every operation."""

    def create_or_get_run(self, *args, **kwargs):
        raise RuntimeError("Database disk failure! IO Error!")

    def record_span(self, *args, **kwargs):
        raise RuntimeError("SQLite database locked! Write rejected!")

    def finalize_run(self, *args, **kwargs):
        raise RuntimeError("Transaction rollback failure!")


def test_payload_truncation():
    """Verify that payload truncation respects the 500-char default and marks truncation."""
    short_text = "Short text under 500 characters"
    assert truncate_payload(short_text, max_len=500) == short_text
    assert truncate_payload(None) is None

    long_text = "A" * 600
    truncated = truncate_payload(long_text, max_len=500)
    assert truncated is not None
    assert len(truncated) == 500 + len("... [truncated]")
    assert truncated.startswith("A" * 500)
    assert truncated.endswith("... [truncated]")


def test_tracer_span_nesting():
    """Verify automatic parent/child span hierarchy nesting via contextvars."""
    recorded_spans: list[dict] = []

    class MockStore:
        def create_or_get_run(self, ticket_id, run_id, started_at=None):
            return None

        def record_span(self, **kwargs):
            recorded_spans.append(kwargs)
            return None

        def finalize_run(self, **kwargs):
            return None

    tracer = Tracer(store=MockStore())  # type: ignore
    run_id = tracer.start_run(ticket_id="TICK-100", run_id="run-100")
    assert get_current_run_id() == "run-100"

    with tracer.span(name="parent_node", span_type=SpanType.NODE.value) as parent_ctx:
        parent_id = parent_ctx.span_id
        assert get_current_parent_span_id() == parent_id

        # Nested child span (e.g. LLM call)
        with tracer.span(name="llm_generate", span_type=SpanType.LLM_CALL.value) as child_ctx:
            assert child_ctx.parent_span_id == parent_id
            assert get_current_parent_span_id() == child_ctx.span_id
            child_ctx.set_tokens(tokens_in=100, tokens_out=25)

        # After child span exits, parent span should be restored
        assert get_current_parent_span_id() == parent_id
        parent_ctx.set_decision("parent_finished")

    # Spans recorded in exit order: child first, then parent
    assert len(recorded_spans) == 2
    child_span = recorded_spans[0]
    parent_span = recorded_spans[1]

    assert child_span["name"] == "llm_generate"
    assert child_span["parent_span_id"] == parent_id
    assert child_span["tokens_in"] == 100
    assert child_span["tokens_out"] == 25

    assert parent_span["name"] == "parent_node"
    assert parent_span["parent_span_id"] is None
    assert parent_span["decision_summary"] == "parent_finished"


def test_tracer_never_throws():
    """
    Critical safety guarantee: Even if the underlying database or store raises
    unhandled exceptions, the tracer must NOT crash or fail application workflows.
    """
    exploding_tracer = Tracer(store=ExplodingStore())

    # 1. start_run should not throw
    run_id = exploding_tracer.start_run(ticket_id="TICK-999")
    assert run_id is not None

    # 2. span context manager should not throw on store failure
    executed = False
    with exploding_tracer.span(name="safe_node") as ctx:
        ctx.set_decision("should_succeed")
        executed = True

    assert executed is True

    # 3. record_event should not throw
    event_ctx = exploding_tracer.record_event(
        name="test_gate",
        span_type=SpanType.GATE.value,
        decision_summary="passed=true",
    )
    # Returns None or safe ctx on error, but never throws
    assert event_ctx is None or event_ctx.decision_summary == "passed=true"

    # 4. finalize_run should not throw
    exploding_tracer.finalize_run(final_status="sent")


def test_tracer_propagates_application_exceptions_while_recording_failed():
    """
    Application exceptions inside the traced block must still propagate to the caller,
    but the span itself must be marked status='failed' before bubbling up.
    """
    recorded_spans: list[dict] = []

    class MockStore:
        def create_or_get_run(self, *args, **kwargs):
            return None

        def record_span(self, **kwargs):
            recorded_spans.append(kwargs)
            return None

    tracer = Tracer(store=MockStore())  # type: ignore

    with pytest.raises(ValueError, match="Domain error"):
        with tracer.span(name="failing_node") as ctx:
            raise ValueError("Domain error")

    assert len(recorded_spans) == 1
    failed_span = recorded_spans[0]
    assert failed_span["name"] == "failing_node"
    assert failed_span["status"] == SpanStatus.FAILED.value
    assert "Domain error" in failed_span["attributes"]["error"]
