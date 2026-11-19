"""
tests/test_triage_node.py

Unit tests for the Triage agent node (triage_node).
Uses FakeLLM to test classification, priority escalation, structured retry recovery,
and call budget exhaustion.
"""

from __future__ import annotations

import json
from app.agents.nodes.triage import triage_node
from app.agents.state import AgentState
from app.db.enums import TicketPriority, TicketStatus
from tests.fakes import FakeLLM


def _make_state(**kwargs: object) -> AgentState:
    state: AgentState = {
        "ticket_id": "TICK-TEST-001",
        "customer_name": "Marcus Vance",
        "channel": "web",
        "subject": "Where is my order?",
        "body": "Order #10821 shipped 3 days ago but no updates.",
        "status": TicketStatus.NEW.value,
        "llm_calls_used": 0,
        "max_llm_calls": 6,
    }
    state.update(kwargs)  # type: ignore[typeddict-item]
    return state


class TestTriageNode:
    def test_triage_happy_path(self) -> None:
        payload = {
            "category": "order_status",
            "priority": "P2",
            "sentiment": "neutral",
            "churn_risk": "low",
            "confidence": 0.94,
            "reasoning": "Standard shipping status inquiry.",
        }
        llm = FakeLLM([json.dumps(payload)])
        state = _make_state()

        result = triage_node(state, llm=llm)

        assert result["category"] == "order_status"
        assert result["priority"] == "P2"
        assert result["sentiment"] == "neutral"
        assert result["churn_risk"] == "low"
        assert result["confidence"] == 0.94
        assert result["triage_reasoning"] == "Standard shipping status inquiry."
        assert result["status"] == TicketStatus.TRIAGING.value
        assert result["llm_calls_used"] == 1

    def test_triage_escalates_to_p1_on_high_churn_risk(self) -> None:
        payload = {
            "category": "order_status",
            "priority": "P3",  # LLM proposed P3, but rule overrides to P1
            "sentiment": "negative",
            "churn_risk": "high",
            "confidence": 0.91,
            "reasoning": "Customer threatening to cancel.",
        }
        llm = FakeLLM([json.dumps(payload)])
        state = _make_state()

        result = triage_node(state, llm=llm)

        assert result["priority"] == TicketPriority.P1.value

    def test_triage_escalates_to_p1_on_angry_refund(self) -> None:
        payload = {
            "category": "refund_billing",
            "priority": "P2",  # LLM proposed P2, but rule overrides to P1
            "sentiment": "angry",
            "churn_risk": "medium",
            "confidence": 0.96,
            "reasoning": "Duplicate billing charge with furious customer.",
        }
        llm = FakeLLM([json.dumps(payload)])
        state = _make_state()

        result = triage_node(state, llm=llm)

        assert result["priority"] == TicketPriority.P1.value

    def test_triage_recovers_after_retry(self) -> None:
        valid_payload = {
            "category": "product_question",
            "priority": "P3",
            "sentiment": "neutral",
            "churn_risk": "low",
            "confidence": 0.90,
            "reasoning": "Product spec question.",
        }
        llm = FakeLLM(["{invalid json", json.dumps(valid_payload)])
        state = _make_state(llm_calls_used=0)

        result = triage_node(state, llm=llm)

        assert result["category"] == "product_question"
        assert result["status"] == TicketStatus.TRIAGING.value
        assert result["llm_calls_used"] == 2  # 1 failure + 1 recovery

    def test_triage_parse_failure_routes_to_untriaged(self) -> None:
        # All 3 attempts return non-JSON
        llm = FakeLLM(["garbage 1", "garbage 2", "garbage 3"])
        state = _make_state(llm_calls_used=0)

        result = triage_node(state, llm=llm)

        assert result["status"] == TicketStatus.UNTRIAGED.value
        assert result["confidence"] == 0.0
        assert result["llm_calls_used"] == 3
        assert "error" in result

    def test_triage_aborts_when_budget_exceeded(self) -> None:
        llm = FakeLLM(["should not be called"])
        state = _make_state(llm_calls_used=6, max_llm_calls=6)

        result = triage_node(state, llm=llm)

        assert result["status"] == "error"
        assert "budget exceeded" in result["error"].lower()
        assert len(llm.calls) == 0
