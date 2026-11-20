"""
tests/test_tickets_api.py

Integration and API tests for /api/tickets endpoints and end-to-end graph execution.
Uses FastAPI TestClient and FakeLLM — completely offline, deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agents.graph import run_ticket_graph
from app.db import session as session_mod
from app.db.enums import TicketStatus
from app.db.models import Base, Draft, Setting, Ticket
from app.main import app
from app.services.kb.retriever import RetrievedChunk
from tests.fakes import FakeLLM


@pytest.fixture(autouse=True)
def setup_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure database tables are initialized and clean in a temporary DB before each test."""
    test_db_path = tmp_path / "test.db"
    test_url = f"sqlite:///{test_db_path}"
    test_engine = create_engine(test_url, connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )

    monkeypatch.setattr("app.db.session.engine", test_engine)
    monkeypatch.setattr("app.db.session.SessionLocal", test_session_local)
    monkeypatch.setattr("app.api.routes.tickets.SessionLocal", test_session_local)

    Base.metadata.create_all(bind=test_engine)
    with test_session_local() as session:
        session.add(Setting(key="auto_send_enabled", value="false"))
        session.commit()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestTicketsAPI:
    def test_create_ticket_endpoint(self, client: TestClient) -> None:
        payload = {
            "channel": "web",
            "customer_name": "Alice Cooper",
            "subject": "Missing items in order #5512",
            "body": "I ordered 3 shirts but only received 2 in the package.",
        }

        # Patch background execution so the API test inspects ticket creation directly
        with patch("app.api.routes.tickets._process_ticket_background") as mock_bg:
            response = client.post("/api/tickets", json=payload)
            assert response.status_code == 202
            data = response.json()
            assert "id" in data
            assert data["id"].startswith("TICK-")
            assert data["status"] == TicketStatus.NEW.value
            mock_bg.assert_called_once_with(data["id"])

    def test_list_and_get_ticket(self, client: TestClient) -> None:
        # Create ticket in DB directly
        with session_mod.SessionLocal() as db:
            ticket = Ticket(
                id="TICK-TEST-LIST-01",
                channel="email",
                customer_name="Bob Dylan",
                subject="Test subject listing",
                body="Test body listing",
                status=TicketStatus.PENDING_REVIEW.value,
                category="order_status",
                priority="P2",
                confidence=0.85,
            )
            draft = Draft(
                ticket_id=ticket.id,
                version=1,
                body="Draft text citing [KB-001]. Best regards, ShopLite Support",
                citations=["KB-001"],
                created_by_node="resolver",
            )
            db.merge(ticket)
            db.merge(draft)
            db.commit()

        # 1. List tickets
        resp = client.get("/api/tickets?category=order_status")
        assert resp.status_code == 200
        list_data = resp.json()
        assert list_data["total"] >= 1
        ticket_ids = [t["id"] for t in list_data["tickets"]]
        assert "TICK-TEST-LIST-01" in ticket_ids

        # 2. Get ticket by ID
        detail_resp = client.get("/api/tickets/TICK-TEST-LIST-01")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["id"] == "TICK-TEST-LIST-01"
        assert detail["customer_name"] == "Bob Dylan"
        assert detail["category"] == "order_status"
        assert len(detail["drafts"]) == 1
        assert detail["drafts"][0]["citations"] == ["KB-001"]

    def test_get_nonexistent_ticket_404(self, client: TestClient) -> None:
        resp = client.get("/api/tickets/TICK-DOES-NOT-EXIST")
        assert resp.status_code == 404

    def test_run_ticket_graph_e2e_with_fakes(self) -> None:
        """
        End-to-end graph test:
        Ticket in DB -> triage (FakeLLM) -> confidence gate -> resolver (FakeLLM + mock KB) -> DB updated.
        """
        ticket_id = "TICK-E2E-FAKE-01"
        with session_mod.SessionLocal() as db:
            t = Ticket(
                id=ticket_id,
                channel="chat",
                customer_name="Charlie Brown",
                subject="Tracking my package",
                body="Where is package #9988?",
                status=TicketStatus.NEW.value,
            )
            db.merge(t)
            db.commit()

        triage_json = json.dumps(
            {
                "category": "order_status",
                "priority": "P2",
                "sentiment": "neutral",
                "churn_risk": "low",
                "confidence": 0.95,
                "reasoning": "Standard package tracking query.",
            }
        )
        resolver_text = (
            "Hi Charlie,\n\n"
            "You can track package #9988 using the link sent to your email [KB-001].\n\n"
            "Best regards,\nShopLite Support"
        )

        llm = FakeLLM([triage_json, resolver_text])

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RetrievedChunk(
                id="KB-001_0",
                article_id="KB-001",
                title="Tracking Info",
                category="shipping",
                chunk_index=0,
                text="Tracking link details.",
                score=0.91,
                content_hash="h1",
                tags=[],
            )
        ]
        mock_retriever.has_kb_gap.return_value = False

        with session_mod.SessionLocal() as db:
            final_state = run_ticket_graph(
                ticket_id=ticket_id,
                db=db,
                llm=llm,
                retriever=mock_retriever,
            )

            # Assert final state
            assert final_state.get("status") == TicketStatus.PENDING_REVIEW.value
            assert final_state.get("category") == "order_status"
            assert final_state.get("priority") == "P2"
            assert final_state.get("citations") == ["KB-001"]
            assert final_state.get("llm_calls_used") == 2

            # Assert database persistence
            refreshed = db.query(Ticket).filter_by(id=ticket_id).first()
            assert refreshed is not None
            assert refreshed.status == TicketStatus.PENDING_REVIEW.value
            assert refreshed.category == "order_status"
            assert len(refreshed.drafts) == 1
            assert refreshed.drafts[0].citations == ["KB-001"]
            assert "[KB-001]" in refreshed.drafts[0].body
