"""
tests/test_approvals_hitl.py

Responsibility: Integration tests for the Critic-Resolver revision loop,
Action Gate (auto-send allowlist vs never-auto-send), LangGraph interrupts,
Human-in-the-Loop approval/edit/reject workflows, and Outbox persistence.
"""

from __future__ import annotations

import json
import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agents.graph import (
    create_agent_graph,
    resume_ticket_graph,
    run_ticket_graph,
)
from app.db import Base, get_db
from app.db.enums import TicketStatus
from app.db.models import ApprovalDecision, Draft, Outbox, Ticket
from app.services.kb.retriever import RetrievedChunk
from app.services.policies.models import PolicyRule, PolicySet
from tests.fakes import FakeLLM


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    testing_session_local = sessionmaker(bind=engine)
    session = testing_session_local()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def memory_checkpointer():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    return SqliteSaver(conn)


@pytest.fixture
def fake_retriever():
    class StubRetriever:
        relevance_threshold = 0.50

        def retrieve(self, query: str, top_k: int = 2) -> list[RetrievedChunk]:
            return [
                RetrievedChunk(
                    id="KB-001_c1",
                    article_id="KB-001",
                    title="Return Policy",
                    category="returns_refunds",
                    chunk_index=0,
                    text="Items can be returned within 30 days.",
                    score=0.92,
                    content_hash="abc",
                )
            ]

        def has_kb_gap(self, chunks: list[RetrievedChunk]) -> bool:
            return False

        def best_score(self, chunks: list[RetrievedChunk]) -> float:
            return 0.92

    return StubRetriever()


@pytest.fixture
def test_policy_set():
    return PolicySet(
        version="test_policy_v1",
        changelog="HITL testing policy",
        rules=[
            PolicyRule(
                id="POL-001",
                title="Refund Limit",
                description="Max refund $20",
                severity="critical",
            ),
            PolicyRule(
                id="POL-004",
                title="No Discounts",
                description="No discounts allowed",
                severity="major",
            ),
            PolicyRule(
                id="POL-007",
                title="Citation Required",
                description="Must cite KB",
                severity="major",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 1. Revision Loop Tests
# ---------------------------------------------------------------------------


def test_revision_loop_improves_draft_to_pass(
    in_memory_db: Session,
    memory_checkpointer: SqliteSaver,
    fake_retriever,
    test_policy_set,
):
    """
    Test revision loop:
    1. Triage classifies ticket as order_status.
    2. Resolver drafts v1 (contains unauthorized discount).
    3. Critic returns 'revise' (major violation POL-004).
    4. Resolver re-drafts with feedback, producing clean draft v2.
    5. Critic returns 'pass'.
    6. Action gate pauses for human approval via interrupt.
    Assert both draft v1 and v2 exist in DB and differ!
    """
    ticket_id = f"T-REV-{uuid.uuid4().hex[:6]}"
    ticket = Ticket(
        id=ticket_id,
        customer_name="John Doe",
        channel="web",
        subject="Where is my order?",
        body="My order #100 is delayed.",
        status=TicketStatus.NEW.value,
    )
    in_memory_db.add(ticket)
    in_memory_db.commit()

    # LLM script:
    # 1. Triage: category order_status, confidence 0.95
    triage_json = json.dumps(
        {
            "category": "order_status",
            "priority": "P2",
            "sentiment": "neutral",
            "churn_risk": "low",
            "confidence": 0.95,
            "reasoning": "Inquiring about order status",
        }
    )
    # 2. Resolver Draft 1: Has discount
    draft_1 = "Your order is delayed. Here is 20% off with coupon SAVE20 per [KB-001]. Best regards, ShopLite Support"
    # 3. Critic 1: Major violation POL-004 -> revise
    critic_1 = json.dumps(
        {
            "verdict": "revise",
            "violations": [
                {
                    "policy_id": "POL-004",
                    "severity": "major",
                    "quoted_span": "20% off with coupon SAVE20",
                    "reason": "Unauthorized discount",
                }
            ],
            "overall_reasoning": "Remove discount coupon.",
        }
    )
    # 4. Resolver Draft 2: Fixed draft without discount
    draft_2 = "Your order is on the way. You can return it within 30 days per [KB-001]. Best regards, ShopLite Support"
    # 5. Critic 2: Pass
    critic_2 = json.dumps(
        {
            "verdict": "pass",
            "violations": [],
            "overall_reasoning": "All violations resolved.",
        }
    )

    fake_llm = FakeLLM([triage_json, draft_1, critic_1, draft_2, critic_2])

    result = run_ticket_graph(
        ticket_id=ticket_id,
        db=in_memory_db,
        llm=fake_llm,
        retriever=fake_retriever,
        checkpointer=memory_checkpointer,
        policy_set=test_policy_set,
        auto_send_override=False,
    )

    # Paused on interrupt for human approval
    assert result.get("status") == TicketStatus.PENDING_REVIEW.value
    assert result.get("critic_verdict") == "pass"

    # Verify both draft versions were stored in DB
    drafts = (
        in_memory_db.query(Draft)
        .filter_by(ticket_id=ticket_id)
        .order_by(Draft.version.asc())
        .all()
    )
    assert len(drafts) == 2
    assert drafts[0].version == 1
    assert "SAVE20" in drafts[0].body
    assert drafts[1].version == 2
    assert "SAVE20" not in drafts[1].body


def test_revision_loop_exhaustion_pends_review(
    in_memory_db: Session,
    memory_checkpointer: SqliteSaver,
    fake_retriever,
    test_policy_set,
):
    """
    Test that if critic requests 'revise' a second time, the revision budget (1)
    is exhausted and the ticket halts at pending_review rather than looping indefinitely.
    """
    ticket_id = f"T-EXH-{uuid.uuid4().hex[:6]}"
    ticket = Ticket(
        id=ticket_id,
        customer_name="Jane",
        channel="web",
        subject="Status",
        body="Order status please",
        status=TicketStatus.NEW.value,
    )
    in_memory_db.add(ticket)
    in_memory_db.commit()

    triage_json = json.dumps(
        {
            "category": "order_status",
            "priority": "P2",
            "sentiment": "neutral",
            "churn_risk": "low",
            "confidence": 0.95,
            "reasoning": "Order status",
        }
    )
    draft_1 = "Bad draft 1 [KB-001]"
    critic_1 = json.dumps(
        {
            "verdict": "revise",
            "violations": [
                {
                    "policy_id": "POL-004",
                    "severity": "major",
                    "quoted_span": "span",
                    "reason": "discount",
                }
            ],
            "overall_reasoning": "Revise 1",
        }
    )
    draft_2 = "Bad draft 2 [KB-001]"
    critic_2 = json.dumps(
        {
            "verdict": "revise",
            "violations": [
                {
                    "policy_id": "POL-004",
                    "severity": "major",
                    "quoted_span": "span",
                    "reason": "still bad",
                }
            ],
            "overall_reasoning": "Revise 2",
        }
    )

    fake_llm = FakeLLM([triage_json, draft_1, critic_1, draft_2, critic_2])

    result = run_ticket_graph(
        ticket_id=ticket_id,
        db=in_memory_db,
        llm=fake_llm,
        retriever=fake_retriever,
        checkpointer=memory_checkpointer,
        policy_set=test_policy_set,
        auto_send_override=False,
    )

    # Critic 2 verdict mapped to escalate because revisions_used == 1 >= max_revision_loops
    assert result.get("critic_verdict") == "escalate"
    assert result.get("status") == TicketStatus.PENDING_REVIEW.value


# ---------------------------------------------------------------------------
# 2. Human-in-the-Loop (Interrupt & Resume) Tests
# ---------------------------------------------------------------------------


def test_hitl_approval_flow(
    in_memory_db: Session,
    memory_checkpointer: SqliteSaver,
    fake_retriever,
    test_policy_set,
):
    """
    Run ticket until interrupt -> call resume_ticket_graph with approve.
    Assert ticket status='sent', Outbox has sent_by='human', ApprovalDecision recorded.
    """
    ticket_id = f"T-HITL-APP-{uuid.uuid4().hex[:6]}"
    ticket = Ticket(
        id=ticket_id,
        customer_name="Bob",
        channel="web",
        subject="Status",
        body="Order status please",
        status=TicketStatus.NEW.value,
    )
    in_memory_db.add(ticket)
    in_memory_db.commit()

    triage_json = json.dumps(
        {
            "category": "order_status",
            "priority": "P2",
            "sentiment": "neutral",
            "churn_risk": "low",
            "confidence": 0.95,
            "reasoning": "status",
        }
    )
    draft = "Your order shipped per [KB-001]. Best regards, ShopLite Support"
    critic = json.dumps(
        {"verdict": "pass", "violations": [], "overall_reasoning": "Good"}
    )

    fake_llm = FakeLLM([triage_json, draft, critic])

    # 1. Run until interrupt
    run_ticket_graph(
        ticket_id=ticket_id,
        db=in_memory_db,
        llm=fake_llm,
        retriever=fake_retriever,
        checkpointer=memory_checkpointer,
        policy_set=test_policy_set,
        auto_send_override=False,
    )

    db_ticket = in_memory_db.query(Ticket).filter_by(id=ticket_id).first()
    assert db_ticket is not None
    assert db_ticket.status == TicketStatus.PENDING_REVIEW.value

    # 2. Human approves
    decision = {"action": "approve"}
    resume_ticket_graph(
        ticket_id=ticket_id,
        decision=decision,
        db=in_memory_db,
        checkpointer=memory_checkpointer,
    )

    in_memory_db.refresh(db_ticket)
    assert db_ticket is not None
    assert db_ticket.status == TicketStatus.SENT.value

    # Verify Outbox
    outbox = in_memory_db.query(Outbox).filter_by(ticket_id=ticket_id).first()
    assert outbox is not None
    assert outbox.sent_by == "human"
    assert "Your order shipped" in outbox.final_body

    # Verify ApprovalDecision audit row
    decision_row = (
        in_memory_db.query(ApprovalDecision).filter_by(ticket_id=ticket_id).first()
    )
    assert decision_row is not None
    assert decision_row.decision == "approve"
    assert decision_row.latency_ms is not None


def test_hitl_edit_flow(
    in_memory_db: Session,
    memory_checkpointer: SqliteSaver,
    fake_retriever,
    test_policy_set,
):
    """
    Human reviewer modifies response before sending.
    Assert final_body in outbox matches the edited body.
    """
    ticket_id = f"T-HITL-EDT-{uuid.uuid4().hex[:6]}"
    ticket = Ticket(
        id=ticket_id,
        customer_name="Carol",
        channel="web",
        subject="Status",
        body="Order status",
        status=TicketStatus.NEW.value,
    )
    in_memory_db.add(ticket)
    in_memory_db.commit()

    triage_json = json.dumps(
        {
            "category": "order_status",
            "priority": "P2",
            "sentiment": "neutral",
            "churn_risk": "low",
            "confidence": 0.95,
            "reasoning": "status",
        }
    )
    draft = "Original AI draft per [KB-001]."
    critic = json.dumps(
        {"verdict": "pass", "violations": [], "overall_reasoning": "Good"}
    )

    fake_llm = FakeLLM([triage_json, draft, critic])

    run_ticket_graph(
        ticket_id=ticket_id,
        db=in_memory_db,
        llm=fake_llm,
        retriever=fake_retriever,
        checkpointer=memory_checkpointer,
        policy_set=test_policy_set,
        auto_send_override=False,
    )

    # Human edits draft
    custom_text = "Hello Carol, I personally verified your order #100 is delivering today. Best, ShopLite Team"
    decision = {"action": "edit", "edited_body": custom_text}
    resume_ticket_graph(
        ticket_id=ticket_id,
        decision=decision,
        db=in_memory_db,
        checkpointer=memory_checkpointer,
    )

    db_ticket = in_memory_db.query(Ticket).filter_by(id=ticket_id).first()
    assert db_ticket is not None
    assert db_ticket.status == TicketStatus.SENT.value

    outbox = in_memory_db.query(Outbox).filter_by(ticket_id=ticket_id).first()
    assert outbox is not None
    assert outbox.final_body == custom_text
    assert outbox.sent_by == "human"

    decision_row = (
        in_memory_db.query(ApprovalDecision).filter_by(ticket_id=ticket_id).first()
    )
    assert decision_row is not None
    assert decision_row.decision == "edit"
    assert decision_row.original_body == draft
    assert decision_row.edited_body == custom_text


def test_hitl_reject_flow(
    in_memory_db: Session,
    memory_checkpointer: SqliteSaver,
    fake_retriever,
    test_policy_set,
):
    """
    Human reviewer rejects response.
    Assert status='rejected', reason recorded, NO outbox row created.
    """
    ticket_id = f"T-HITL-REJ-{uuid.uuid4().hex[:6]}"
    ticket = Ticket(
        id=ticket_id,
        customer_name="Dan",
        channel="web",
        subject="Status",
        body="Order status",
        status=TicketStatus.NEW.value,
    )
    in_memory_db.add(ticket)
    in_memory_db.commit()

    triage_json = json.dumps(
        {
            "category": "order_status",
            "priority": "P2",
            "sentiment": "neutral",
            "churn_risk": "low",
            "confidence": 0.95,
            "reasoning": "status",
        }
    )
    draft = "Draft to reject [KB-001]."
    critic = json.dumps(
        {"verdict": "pass", "violations": [], "overall_reasoning": "Good"}
    )

    fake_llm = FakeLLM([triage_json, draft, critic])

    run_ticket_graph(
        ticket_id=ticket_id,
        db=in_memory_db,
        llm=fake_llm,
        retriever=fake_retriever,
        checkpointer=memory_checkpointer,
        policy_set=test_policy_set,
        auto_send_override=False,
    )

    decision = {"action": "reject", "reason": "Customer needs fraud verification first"}
    resume_ticket_graph(
        ticket_id=ticket_id,
        decision=decision,
        db=in_memory_db,
        checkpointer=memory_checkpointer,
    )

    db_ticket = in_memory_db.query(Ticket).filter_by(id=ticket_id).first()
    assert db_ticket is not None
    assert db_ticket.status == TicketStatus.REJECTED.value

    # No outbox row
    outbox = in_memory_db.query(Outbox).filter_by(ticket_id=ticket_id).first()
    assert outbox is None

    decision_row = (
        in_memory_db.query(ApprovalDecision).filter_by(ticket_id=ticket_id).first()
    )
    assert decision_row is not None
    assert decision_row.decision == "reject"
    assert decision_row.reason == "Customer needs fraud verification first"


# ---------------------------------------------------------------------------
# 3. Action Gate & Auto-Send Eligibility Tests
# ---------------------------------------------------------------------------


def test_auto_send_enabled_clean_ticket(
    in_memory_db: Session,
    memory_checkpointer: SqliteSaver,
    fake_retriever,
    test_policy_set,
):
    """
    Clean order_status ticket auto-sends when auto_send_enabled=True.
    """
    ticket_id = f"T-AUTO-{uuid.uuid4().hex[:6]}"
    ticket = Ticket(
        id=ticket_id,
        customer_name="Auto User",
        channel="web",
        subject="Tracking",
        body="Where is package?",
        status=TicketStatus.NEW.value,
    )
    in_memory_db.add(ticket)
    in_memory_db.commit()

    triage_json = json.dumps(
        {
            "category": "order_status",
            "priority": "P2",
            "sentiment": "neutral",
            "churn_risk": "low",
            "confidence": 0.95,
            "reasoning": "status",
        }
    )
    draft = "Tracking info per [KB-001]."
    critic = json.dumps(
        {"verdict": "pass", "violations": [], "overall_reasoning": "Good"}
    )

    fake_llm = FakeLLM([triage_json, draft, critic])

    result = run_ticket_graph(
        ticket_id=ticket_id,
        db=in_memory_db,
        llm=fake_llm,
        retriever=fake_retriever,
        checkpointer=memory_checkpointer,
        policy_set=test_policy_set,
        auto_send_override=True,  # Master switch ON
    )

    assert result.get("status") == TicketStatus.AUTO_SENT.value

    outbox = in_memory_db.query(Outbox).filter_by(ticket_id=ticket_id).first()
    assert outbox is not None
    assert outbox.sent_by == "auto"


def test_never_auto_send_category_always_pends_even_if_auto_send_enabled(
    in_memory_db: Session,
    memory_checkpointer: SqliteSaver,
    fake_retriever,
    test_policy_set,
):
    """
    refund_billing is in never_auto_send: even with auto_send_enabled=True,
    ticket MUST pend review via interrupt!
    """
    ticket_id = f"T-NEVER-{uuid.uuid4().hex[:6]}"
    ticket = Ticket(
        id=ticket_id,
        customer_name="Refund User",
        channel="web",
        subject="Refund request",
        body="I want a refund for damaged item",
        status=TicketStatus.NEW.value,
    )
    in_memory_db.add(ticket)
    in_memory_db.commit()

    triage_json = json.dumps(
        {
            "category": "refund_billing",
            "priority": "P2",
            "sentiment": "neutral",
            "churn_risk": "low",
            "confidence": 0.95,
            "reasoning": "refund",
        }
    )
    draft = "We received your refund inquiry per [KB-001]."
    critic = json.dumps(
        {"verdict": "pass", "violations": [], "overall_reasoning": "Clean draft"}
    )

    fake_llm = FakeLLM([triage_json, draft, critic])

    result = run_ticket_graph(
        ticket_id=ticket_id,
        db=in_memory_db,
        llm=fake_llm,
        retriever=fake_retriever,
        checkpointer=memory_checkpointer,
        policy_set=test_policy_set,
        auto_send_override=True,  # Master switch ON
    )

    # Must NOT auto-send because refund_billing is in never_auto_send
    assert result.get("status") == TicketStatus.PENDING_REVIEW.value
    outbox = in_memory_db.query(Outbox).filter_by(ticket_id=ticket_id).first()
    assert outbox is None


# ---------------------------------------------------------------------------
# 4. REST API Endpoint Tests
# ---------------------------------------------------------------------------


def test_approvals_api_endpoints(
    in_memory_db: Session,
    memory_checkpointer: SqliteSaver,
    fake_retriever,
    test_policy_set,
):
    """Test GET /api/approvals, POST /api/approvals/{id}/decision, and GET /api/outbox."""
    from fastapi import FastAPI
    from app.api.routes import (
        approvals as approvals_route,
        outbox as outbox_route,
        policies as policies_route,
    )
    from app.models.approvals import ApprovalDecisionRequest

    test_app = FastAPI()
    test_app.include_router(approvals_route.router, prefix="/api")
    test_app.include_router(outbox_route.router, prefix="/api")
    test_app.include_router(policies_route.router, prefix="/api")
    test_app.dependency_overrides[get_db] = lambda: in_memory_db

    # Create ticket in pending_review with draft and critic verdict
    ticket_id = f"T-API-{uuid.uuid4().hex[:6]}"
    ticket = Ticket(
        id=ticket_id,
        customer_name="API User",
        channel="web",
        subject="API Test Ticket",
        body="Testing API flow",
        status=TicketStatus.PENDING_REVIEW.value,
        category="order_status",
        priority="P2",
        confidence=0.90,
        critic_verdict="pass",
        critic_reasoning="Verified clean",
    )
    draft = Draft(
        ticket_id=ticket_id,
        version=1,
        body="API draft text [KB-001]",
        citations=["KB-001"],
        created_by_node="resolver",
    )
    in_memory_db.add(ticket)
    in_memory_db.add(draft)
    in_memory_db.commit()

    triage_json = json.dumps(
        {
            "category": "order_status",
            "priority": "P2",
            "sentiment": "neutral",
            "churn_risk": "low",
            "confidence": 0.90,
            "reasoning": "status",
        }
    )
    draft_body = "API draft text [KB-001]"
    critic_json = json.dumps(
        {"verdict": "pass", "violations": [], "overall_reasoning": "Good"}
    )
    fake_llm = FakeLLM([triage_json, draft_body, critic_json])

    # Pre-populate graph checkpoint so resume works
    graph = create_agent_graph(
        llm=fake_llm,
        retriever=fake_retriever,
        checkpointer=memory_checkpointer,
        policy_set=test_policy_set,
    )
    config = {"configurable": {"thread_id": ticket_id}}
    graph.invoke(
        {
            "ticket_id": ticket_id,
            "status": TicketStatus.PENDING_REVIEW.value,
            "draft_body": "API draft text [KB-001]",
            "category": "order_status",
            "critic_verdict": "pass",
            "auto_send_enabled": False,
        },
        config=config,
    )

    # 1. GET /api/approvals handler
    approvals = approvals_route.list_pending_approvals(db=in_memory_db)
    assert any(a.ticket_id == ticket_id for a in approvals)

    # 2. POST /api/approvals/{ticket_id}/decision handler
    from app.agents import graph as graph_mod

    original_get_checkpointer = graph_mod.get_checkpointer
    graph_mod.get_checkpointer = lambda *args, **kwargs: memory_checkpointer

    try:
        dec_res = approvals_route.submit_approval_decision(
            ticket_id=ticket_id,
            decision_req=ApprovalDecisionRequest(action="approve"),
            db=in_memory_db,
        )
        assert dec_res.ticket_id == ticket_id
        assert dec_res.status == "sent"
        assert dec_res.action == "approve"

        # 3. GET /api/outbox handler
        outbox_items = outbox_route.list_outbox_entries(db=in_memory_db)
        assert any(item.ticket_id == ticket_id for item in outbox_items)

        # 4. GET /api/policies handler
        pol_res = policies_route.get_policies()
        assert pol_res.version == "shoplite_v1"
        assert len(pol_res.rules) >= 8

    finally:
        graph_mod.get_checkpointer = original_get_checkpointer
