"""
tests/test_graph_gates.py

Unit tests for pure conditional routing gates and confidence abstention logic.
"""

from __future__ import annotations

from app.agents.graph import confidence_abstain_node, route_after_triage
from app.agents.state import AgentState
from app.db.enums import TicketStatus


class TestGraphGates:
    def test_route_after_triage_untriaged(self) -> None:
        state: AgentState = {
            "ticket_id": "TICK-GATE-01",
            "status": TicketStatus.UNTRIAGED.value,
            "confidence": 0.0,
        }
        assert route_after_triage(state) == "untriaged"

    def test_route_after_triage_low_confidence(self) -> None:
        # threshold is 0.70; 0.65 should trigger low_confidence
        state: AgentState = {
            "ticket_id": "TICK-GATE-02",
            "status": TicketStatus.TRIAGING.value,
            "confidence": 0.65,
        }
        assert route_after_triage(state) == "low_confidence"

    def test_route_after_triage_none_confidence(self) -> None:
        state: AgentState = {
            "ticket_id": "TICK-GATE-03",
            "status": TicketStatus.TRIAGING.value,
            "confidence": None,
        }
        assert route_after_triage(state) == "low_confidence"

    def test_route_after_triage_grounded_high_confidence(self) -> None:
        state: AgentState = {
            "ticket_id": "TICK-GATE-04",
            "status": TicketStatus.TRIAGING.value,
            "confidence": 0.85,
        }
        assert route_after_triage(state) == "resolver"

    def test_confidence_abstain_node(self) -> None:
        state: AgentState = {
            "ticket_id": "TICK-GATE-05",
            "status": TicketStatus.TRIAGING.value,
            "confidence": 0.40,
        }
        update = confidence_abstain_node(state)
        assert update["status"] == TicketStatus.PENDING_REVIEW.value
        assert update["draft_body"] is None
        assert update["citations"] == []
