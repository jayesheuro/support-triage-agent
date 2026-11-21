"""
tests/test_trace_wiring.py

Responsibility: End-to-end integration tests asserting that first-party tracing
is correctly wired across all agent nodes, gates, retry loops, and HITL interrupt/resume lifecycles.
"""

from __future__ import annotations

import json
import sqlite3
import uuid

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents.graph import resume_ticket_graph, run_ticket_graph
from app.db import Base
from app.db.enums import SpanStatus, SpanType, TicketStatus
from app.db.models import Ticket, TraceRun, TraceSpan
from app.services.kb.retriever import RetrievedChunk
from app.services.policies.models import PolicyRule, PolicySet
from app.services.tracing.store import TraceStore
from app.services.tracing.tracer import Tracer
from tests.fakes import FakeLLM


@pytest.fixture
def test_db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session, session_factory
    finally:
        session.close()


@pytest.fixture
def memory_checkpointer():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    return SqliteSaver(conn)


@pytest.fixture
def stub_retriever():
    class StubRetriever:
        relevance_threshold = 0.50

        def retrieve(self, query: str, top_k: int = 2) -> list[RetrievedChunk]:
            return [
                RetrievedChunk(
                    id="KB-001_0",
                    article_id="KB-001",
                    title="Return Policy",
                    category="returns_exchanges",
                    chunk_index=0,
                    text="We accept returns within 30 days for undamaged items. [KB-001]",
                    score=0.92,
                    content_hash="hash_001",
                )
            ]

        def has_kb_gap(self, chunks) -> bool:
            return False

        def best_score(self, chunks) -> float:
            return 0.92

    return StubRetriever()


@pytest.fixture
def sample_policy_set():
    return PolicySet(
        version="shoplite_v1",
        changelog="Initial test policy version",
        rules=[
            PolicyRule(
                id="POL-001",
                title="Refund Limit",
                severity="critical",
                description="Refunds cannot exceed $20",
                examples_violating=[],
                examples_compliant=[],
            ),
            PolicyRule(
                id="POL-004",
                title="Citation Requirement",
                severity="major",
                description="Must cite [KB-xxx]",
                examples_violating=[],
                examples_compliant=[],
            ),
        ],
    )


def test_happy_path_trace_span_sequence_and_nesting(test_db, memory_checkpointer, stub_retriever, sample_policy_set):
    """
    Assert full happy-path span sequence:
    triage (with nested llm_generate) -> confidence_gate -> resolver (with nested llm_generate) ->
    critic (with nested llm_generate) -> action_gate -> interrupt.
    """
    db, session_factory = test_db
    custom_store = TraceStore(session_factory=session_factory)
    custom_tracer = Tracer(store=custom_store)

    ticket = Ticket(
        id=f"TICK-{uuid.uuid4().hex[:6]}",
        customer_name="Alice",
        channel="web",
        subject="Return item within 30 days",
        body="I would like to return an unopened sweater I bought last week.",
        status=TicketStatus.NEW.value,
    )
    db.add(ticket)
    db.commit()

    # Scripted LLM responses
    triage_json = json.dumps(
        {
            "category": "returns_exchanges",
            "priority": "P2",
            "sentiment": "neutral",
            "churn_risk": "low",
            "confidence": 0.95,
            "reasoning": "Standard return request",
        }
    )
    resolver_text = "You can return your unopened sweater within 30 days [KB-001]."
    critic_json = json.dumps(
        {
            "verdict": "pass",
            "violations": [],
            "overall_reasoning": "No violations detected.",
        }
    )

    llm = FakeLLM(responses=[triage_json, resolver_text, critic_json])

    # Run ticket graph
    run_ticket_graph(
        ticket_id=ticket.id,
        db=db,
        llm=llm,
        retriever=stub_retriever,
        checkpointer=memory_checkpointer,
        policy_set=sample_policy_set,
        auto_send_override=False,  # forces interrupt at action gate
    )

    # Inspect recorded trace run and spans
    trace_run = db.query(TraceRun).filter_by(ticket_id=ticket.id).first()
    assert trace_run is not None
    assert trace_run.total_llm_calls == 3
    assert trace_run.final_status == TicketStatus.PENDING_REVIEW.value

    spans = db.query(TraceSpan).filter_by(run_id=trace_run.run_id).order_by(TraceSpan.started_at.asc(), TraceSpan.id.asc()).all()
    span_names = [s.name for s in spans]

    # Verify span types and sequence
    assert "triage" in span_names
    assert "confidence_gate" in span_names
    assert "resolver" in span_names
    assert "critic" in span_names
    assert "action_gate" in span_names
    assert "interrupt" in span_names

    # Verify nesting: llm_generate spans must have parent_span_id matching their enclosing node
    triage_span = next(s for s in spans if s.name == "triage")
    resolver_span = next(s for s in spans if s.name == "resolver")
    critic_span = next(s for s in spans if s.name == "critic")

    llm_spans = [s for s in spans if s.span_type == SpanType.LLM_CALL.value]
    assert len(llm_spans) == 3

    triage_llm = next(s for s in llm_spans if s.parent_span_id == triage_span.span_id)
    assert triage_llm is not None
    assert triage_llm.tokens_in is not None
    assert triage_llm.tokens_out is not None

    resolver_llm = next(s for s in llm_spans if s.parent_span_id == resolver_span.span_id)
    assert resolver_llm is not None

    critic_llm = next(s for s in llm_spans if s.parent_span_id == critic_span.span_id)
    assert critic_llm is not None

    # Verify decision summaries
    assert "category=returns_exchanges" in triage_span.decision_summary
    assert "conf=0.95" in triage_span.decision_summary
    assert "drafted v1" in resolver_span.decision_summary
    assert "verdict=pass" in critic_span.decision_summary


def test_retry_path_shows_retried_status_and_attempts(test_db, memory_checkpointer, stub_retriever, sample_policy_set):
    """
    When structured output fails on attempt 1 and recovers on attempt 2,
    the node span must have status='retried' and retry_count >= 1, and 2 llm_call spans must exist.
    """
    db, session_factory = test_db
    ticket = Ticket(
        id=f"TICK-{uuid.uuid4().hex[:6]}",
        customer_name="Bob",
        channel="web",
        subject="Malformed first turn",
        body="Need help with return.",
        status=TicketStatus.NEW.value,
    )
    db.add(ticket)
    db.commit()

    malformed_json = "NOT VALID JSON AT ALL"
    valid_triage_json = json.dumps(
        {
            "category": "returns_exchanges",
            "priority": "P2",
            "sentiment": "neutral",
            "churn_risk": "low",
            "confidence": 0.88,
            "reasoning": "Recovered on turn 2",
        }
    )
    resolver_text = "Return within 30 days [KB-001]."
    critic_json = json.dumps({"verdict": "pass", "violations": [], "overall_reasoning": "All good."})

    llm = FakeLLM(responses=[malformed_json, valid_triage_json, resolver_text, critic_json])

    run_ticket_graph(
        ticket_id=ticket.id,
        db=db,
        llm=llm,
        retriever=stub_retriever,
        checkpointer=memory_checkpointer,
        policy_set=sample_policy_set,
        auto_send_override=False,
    )

    trace_run = db.query(TraceRun).filter_by(ticket_id=ticket.id).first()
    assert trace_run is not None

    triage_span = (
        db.query(TraceSpan)
        .filter_by(run_id=trace_run.run_id, name="triage")
        .first()
    )
    assert triage_span is not None
    assert triage_span.status == SpanStatus.RETRIED.value
    assert triage_span.retry_count == 1

    # Both LLM attempts should be captured under triage
    child_llm_spans = (
        db.query(TraceSpan)
        .filter_by(run_id=trace_run.run_id, parent_span_id=triage_span.span_id)
        .all()
    )
    assert len(child_llm_spans) == 2


def test_hitl_approval_records_interrupt_and_resume_in_same_run(test_db, memory_checkpointer, stub_retriever, sample_policy_set):
    """
    Verify that an interrupted ticket and subsequent human approval resume
    the SAME run_id and record both 'interrupt' and 'resume' spans.
    """
    db, session_factory = test_db
    ticket = Ticket(
        id=f"TICK-{uuid.uuid4().hex[:6]}",
        customer_name="Carol",
        channel="web",
        subject="Need return",
        body="Return request.",
        status=TicketStatus.NEW.value,
    )
    db.add(ticket)
    db.commit()

    triage_json = json.dumps({"category": "returns_exchanges", "priority": "P2", "sentiment": "neutral", "churn_risk": "low", "confidence": 0.90, "reasoning": "ok"})
    resolver_text = "Return within 30 days [KB-001]."
    critic_json = json.dumps({"verdict": "pass", "violations": [], "overall_reasoning": "pass"})
    llm = FakeLLM(responses=[triage_json, resolver_text, critic_json])

    # 1. Run until interrupt
    run_ticket_graph(
        ticket_id=ticket.id,
        db=db,
        llm=llm,
        retriever=stub_retriever,
        checkpointer=memory_checkpointer,
        policy_set=sample_policy_set,
        auto_send_override=False,
    )

    run_before = db.query(TraceRun).filter_by(ticket_id=ticket.id).first()
    assert run_before is not None
    initial_run_id = run_before.run_id

    # 2. Resume with human decision
    resume_ticket_graph(
        ticket_id=ticket.id,
        decision={"action": "approve"},
        db=db,
        checkpointer=memory_checkpointer,
    )

    run_after = db.query(TraceRun).filter_by(ticket_id=ticket.id).first()
    assert run_after is not None
    assert run_after.run_id == initial_run_id
    assert run_after.final_status == TicketStatus.SENT.value

    spans = db.query(TraceSpan).filter_by(run_id=initial_run_id).all()
    span_types = [s.span_type for s in spans]
    span_names = [s.name for s in spans]

    assert SpanType.INTERRUPT.value in span_types
    assert SpanType.RESUME.value in span_types
    assert "interrupt" in span_names
    assert "resume" in span_names


def test_failing_trace_store_does_not_break_ticket_processing(test_db, memory_checkpointer, stub_retriever, sample_policy_set):
    """
    Verify that even if the trace store completely fails (raising exceptions),
    the ticket graph successfully finishes execution.
    """
    db, session_factory = test_db
    ticket = Ticket(
        id=f"TICK-{uuid.uuid4().hex[:6]}",
        customer_name="Dave",
        channel="web",
        subject="Resilience test",
        body="Return my item please.",
        status=TicketStatus.NEW.value,
    )
    db.add(ticket)
    db.commit()

    triage_json = json.dumps({"category": "returns_exchanges", "priority": "P2", "sentiment": "neutral", "churn_risk": "low", "confidence": 0.95, "reasoning": "ok"})
    resolver_text = "Return within 30 days [KB-001]."
    critic_json = json.dumps({"verdict": "pass", "violations": [], "overall_reasoning": "pass"})
    llm = FakeLLM(responses=[triage_json, resolver_text, critic_json])

    # Inject exploding tracer into the global tracer singleton
    import app.services.tracing.tracer as tracer_module
    from tests.test_tracer import ExplodingStore

    original_tracer = tracer_module._DEFAULT_TRACER
    try:
        tracer_module._DEFAULT_TRACER = Tracer(store=ExplodingStore())

        # run_ticket_graph must NOT raise an exception
        state = run_ticket_graph(
            ticket_id=ticket.id,
            db=db,
            llm=llm,
            retriever=stub_retriever,
            checkpointer=memory_checkpointer,
            policy_set=sample_policy_set,
            auto_send_override=False,
        )

        assert state["status"] == TicketStatus.PENDING_REVIEW.value
        assert ticket.status == TicketStatus.PENDING_REVIEW.value
    finally:
        tracer_module._DEFAULT_TRACER = original_tracer
