"""
tests/test_traces_api.py

Responsibility: Integration and API endpoint tests for GET /api/tickets/{id}/trace
and GET /api/traces/stats.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import traces as traces_route
from app.core.errors import register_exception_handlers
from app.db import Base, get_db
from app.db.enums import SpanStatus, SpanType, TicketStatus
from app.db.models import Ticket, TraceRun, TraceSpan
from app.services.tracing.store import TraceStore
import app.services.tracing.store as store_module


@pytest.fixture
def api_test_db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    custom_store = TraceStore(session_factory=session_factory)
    orig_store = store_module._DEFAULT_TRACE_STORE
    store_module._DEFAULT_TRACE_STORE = custom_store

    test_app = FastAPI()
    register_exception_handlers(test_app)
    test_app.include_router(traces_route.router, prefix="/api")

    def override_get_db():
        with session_factory() as s:
            yield s

    test_app.dependency_overrides[get_db] = override_get_db

    try:
        yield session, test_app
    finally:
        session.close()
        store_module._DEFAULT_TRACE_STORE = orig_store


@pytest.fixture
def client(api_test_db) -> TestClient:
    _, test_app = api_test_db
    return TestClient(test_app)


def test_get_ticket_trace_404_for_missing_ticket(client: TestClient):
    """Assert 404 is returned when ticket does not exist."""
    response = client.get("/api/tickets/TICK-NONEXISTENT/trace")
    assert response.status_code == 404
    assert "not found" in response.text.lower()


def test_get_ticket_trace_empty_runs_for_untraced_ticket(client: TestClient, api_test_db):
    """404-safe: ticket processed before tracing returns empty runs list."""
    db, _ = api_test_db
    ticket_id = f"TICK-{uuid.uuid4().hex[:6]}"
    ticket = Ticket(
        id=ticket_id,
        customer_name="Old Customer",
        channel="web",
        subject="Old Ticket",
        body="Processed before tracing existed.",
        status=TicketStatus.SENT.value,
    )
    db.add(ticket)
    db.commit()

    response = client.get(f"/api/tickets/{ticket_id}/trace")
    assert response.status_code == 200
    data = response.json()
    assert data["ticket_id"] == ticket_id
    assert data["runs"] == []


def test_get_ticket_trace_returns_ordered_spans(client: TestClient, api_test_db):
    """Assert that trace endpoint returns all runs and ordered nested spans."""
    db, _ = api_test_db
    ticket_id = f"TICK-{uuid.uuid4().hex[:6]}"
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    ticket = Ticket(
        id=ticket_id,
        customer_name="Alice",
        channel="web",
        subject="Return item",
        body="I want to return an item.",
        status=TicketStatus.SENT.value,
    )
    db.add(ticket)

    run = TraceRun(
        run_id=run_id,
        ticket_id=ticket_id,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        total_duration_ms=450.0,
        total_llm_calls=2,
        total_tokens_in=150,
        total_tokens_out=60,
        final_status=TicketStatus.SENT.value,
    )
    db.add(run)

    span1 = TraceSpan(
        run_id=run_id,
        span_id="span-1",
        name="triage",
        span_type=SpanType.NODE.value,
        duration_ms=200.0,
        status=SpanStatus.OK.value,
        prompt_name="triage",
        prompt_version="1.0.0",
        decision_summary="category=returns_exchanges conf=0.92",
    )
    span2 = TraceSpan(
        run_id=run_id,
        span_id="span-2",
        parent_span_id="span-1",
        name="llm_generate",
        span_type=SpanType.LLM_CALL.value,
        duration_ms=195.0,
        status=SpanStatus.OK.value,
        model="llama3.2:latest",
        tokens_in=100,
        tokens_out=40,
        decision_summary="model=llama3.2:latest in=100 out=40",
    )
    db.add_all([span1, span2])
    db.commit()

    response = client.get(f"/api/tickets/{ticket_id}/trace")
    assert response.status_code == 200
    data = response.json()

    assert data["ticket_id"] == ticket_id
    assert len(data["runs"]) == 1
    run_data = data["runs"][0]
    assert run_data["run_id"] == run_id
    assert run_data["total_llm_calls"] == 2
    assert len(run_data["spans"]) == 2

    # Check child span
    child = next(s for s in run_data["spans"] if s["span_id"] == "span-2")
    assert child["parent_span_id"] == "span-1"
    assert child["tokens_in"] == 100
    assert child["tokens_out"] == 40


def test_get_trace_stats_aggregation(client: TestClient, api_test_db):
    """Seed test data and verify aggregate calculations for dashboard metrics."""
    db, _ = api_test_db
    ticket1_id = "TICK-STATS-1"
    ticket2_id = "TICK-STATS-2"

    for tid in (ticket1_id, ticket2_id):
        db.add(
            Ticket(
                id=tid,
                customer_name="User",
                channel="web",
                subject="Test",
                body="Test ticket.",
                status=TicketStatus.SENT.value,
            )
        )

    run1 = TraceRun(
        run_id="run-s1",
        ticket_id=ticket1_id,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        total_duration_ms=500.0,
        total_llm_calls=2,
        total_tokens_in=200,
        total_tokens_out=50,
        final_status=TicketStatus.AUTO_SENT.value,
    )
    run2 = TraceRun(
        run_id="run-s2",
        ticket_id=ticket2_id,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        total_duration_ms=600.0,
        total_llm_calls=3,
        total_tokens_in=300,
        total_tokens_out=70,
        final_status=TicketStatus.SENT.value,
    )
    db.add_all([run1, run2])

    # Seed spans
    spans = [
        TraceSpan(
            run_id="run-s1",
            span_id="s1-1",
            name="triage",
            span_type=SpanType.NODE.value,
            duration_ms=100.0,
            status=SpanStatus.OK.value,
            retry_count=0,
            decision_summary="category=returns_exchanges conf=0.90",
        ),
        TraceSpan(
            run_id="run-s2",
            span_id="s2-1",
            name="triage",
            span_type=SpanType.NODE.value,
            duration_ms=200.0,
            status=SpanStatus.RETRIED.value,
            retry_count=1,
            decision_summary="category=refund_billing conf=0.85",
        ),
        TraceSpan(
            run_id="run-s1",
            span_id="s1-2",
            name="critic",
            span_type=SpanType.NODE.value,
            duration_ms=150.0,
            status=SpanStatus.OK.value,
            decision_summary="verdict=pass violations=0",
        ),
        TraceSpan(
            run_id="run-s2",
            span_id="s2-2",
            name="critic",
            span_type=SpanType.NODE.value,
            duration_ms=180.0,
            status=SpanStatus.OK.value,
            decision_summary="verdict=revise violations=1",
        ),
        TraceSpan(
            run_id="run-s1",
            span_id="s1-3",
            name="resolver",
            span_type=SpanType.NODE.value,
            duration_ms=250.0,
            status=SpanStatus.OK.value,
            decision_summary="drafted v1 citations=[KB-001]",
        ),
        TraceSpan(
            run_id="run-s2",
            span_id="s2-3",
            name="resolver",
            span_type=SpanType.NODE.value,
            duration_ms=220.0,
            status=SpanStatus.OK.value,
            decision_summary="kb_gap=True best_score=0.41",
        ),
    ]
    db.add_all(spans)
    db.commit()

    response = client.get("/api/traces/stats?window=24h")
    assert response.status_code == 200
    stats = response.json()

    assert stats["total_tickets_traced"] == 2
    assert stats["total_runs"] == 2
    assert stats["total_llm_calls"] == 5
    assert stats["tokens"]["prompt_tokens"] == 500
    assert stats["tokens"]["completion_tokens"] == 120
    assert stats["tokens"]["total_tokens"] == 620
    assert stats["avg_tokens_per_ticket"] == 310.0

    # Pricing estimate
    assert stats["estimated_cost_usd"] > 0.0

    # Node latency averages
    assert "triage" in stats["node_durations"]
    assert stats["node_durations"]["triage"]["avg_ms"] == 150.0
    assert stats["node_durations"]["triage"]["count"] == 2

    # Retry rate on triage: 1 out of 2 retried = 0.50
    assert stats["retry_rates"]["triage"] == 0.5

    # KB gap rate: 1 out of 2 resolver runs had kb_gap = 0.50
    assert stats["kb_gap_rate"] == 0.5

    # Critic verdicts: 1 pass, 1 revise, 0 escalate
    assert stats["verdict_distribution"]["pass"] == 1
    assert stats["verdict_distribution"]["revise"] == 1
    assert stats["verdict_distribution"]["escalate"] == 0

    # Dispatch stats
    assert stats["dispatch_stats"]["auto_sent"] == 1
    assert stats["dispatch_stats"]["human_sent"] == 1
    assert stats["dispatch_stats"]["auto_send_ratio"] == 0.5


def test_invalid_stats_window_rejected(client: TestClient):
    """Assert ValidationError (400) on unsupported window values."""
    response = client.get("/api/traces/stats?window=invalid_window")
    assert response.status_code == 400
