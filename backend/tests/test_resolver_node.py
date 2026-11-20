"""
tests/test_resolver_node.py

Unit tests for the Resolver agent node (resolver_node).
Tests grounded generation with citation validation, kb_gap abstention,
hallucinated-citation retry, and budget caps.
Uses FakeLLM and a lightweight mock retriever — zero external dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.agents.nodes.resolver import extract_citations, resolver_node
from app.agents.state import AgentState
from app.db.enums import TicketStatus
from app.services.kb.retriever import RetrievedChunk
from tests.fakes import FakeLLM


def _make_chunk(article_id: str, title: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{article_id}_0",
        article_id=article_id,
        title=title,
        category="shipping",
        chunk_index=0,
        text=text,
        score=score,
        content_hash="mock_hash",
        tags=[],
    )


def _make_state(**kwargs: object) -> AgentState:
    state: AgentState = {
        "ticket_id": "TICK-RES-001",
        "customer_name": "Elena Rostova",
        "channel": "email",
        "subject": "Delivery timeline inquiry",
        "body": "When will order #11450 arrive with expedited shipping?",
        "category": "order_status",
        "priority": "P2",
        "status": TicketStatus.TRIAGING.value,
        "llm_calls_used": 1,
        "max_llm_calls": 6,
    }
    state.update(kwargs)  # type: ignore[typeddict-item]
    return state


class TestExtractCitations:
    def test_extract_citations(self) -> None:
        text = "According to [KB-001] and [KB-012], standard shipping takes 3-5 days [KB-001]."
        citations = extract_citations(text)
        assert citations == ["KB-001", "KB-012"]

    def test_empty_citations(self) -> None:
        assert extract_citations("No articles cited here.") == []


class TestResolverNode:
    def test_resolver_happy_path(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_chunk("KB-005", "Expedited Shipping", "Expedited shipping takes 2 business days.", 0.85),
            _make_chunk("KB-006", "Order Tracking", "Tracking is provided via email.", 0.72),
        ]
        mock_retriever.has_kb_gap.return_value = False

        llm_reply = (
            "Hello Elena,\n\n"
            "Your expedited order will arrive within 2 business days as stated in [KB-005]. "
            "You can track its progress via the email link [KB-006].\n\n"
            "Best regards,\nShopLite Support"
        )
        llm = FakeLLM([llm_reply])
        state = _make_state(llm_calls_used=1)

        result = resolver_node(state, llm=llm, retriever=mock_retriever)

        assert result["kb_gap"] is False
        assert result["status"] == TicketStatus.PENDING_REVIEW.value
        assert result["citations"] == ["KB-005", "KB-006"]
        assert "Elena" in result["draft_body"]
        assert result["llm_calls_used"] == 2  # 1 from state + 1 resolver call

    def test_resolver_kb_gap_abstention(self) -> None:
        """
        When best retrieval score is below threshold, resolver sets kb_gap=True,
        generates NO draft, and routes to pending_review without invoking LLM.
        """
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_chunk("KB-099", "Unrelated Article", "Something completely unrelated.", 0.28)
        ]
        mock_retriever.has_kb_gap.return_value = True
        mock_retriever.best_score.return_value = 0.28
        mock_retriever.relevance_threshold = 0.50

        llm = FakeLLM(["Should never be called"])
        state = _make_state(
            subject="Vintage bicycle repair?",
            body="Do you tune 1980s bicycles in Chicago?",
            llm_calls_used=1,
        )

        result = resolver_node(state, llm=llm, retriever=mock_retriever)

        assert result["kb_gap"] is True
        assert result["draft_body"] is None
        assert result["citations"] == []
        assert result["status"] == TicketStatus.PENDING_REVIEW.value
        assert len(llm.calls) == 0  # Zero LLM calls made on abstention

    def test_resolver_retries_on_hallucinated_citation(self) -> None:
        """
        Attempt 1: Draft cites [KB-999] (not in retrieved set [KB-005])
        Attempt 2: Retry with correction cites [KB-005] correctly
        """
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_chunk("KB-005", "Expedited Shipping", "Expedited shipping takes 2 days.", 0.88)
        ]
        mock_retriever.has_kb_gap.return_value = False

        hallucinated_draft = "According to our policy [KB-999], it takes 2 days. Best regards, ShopLite Support"
        corrected_draft = "According to our expedited policy [KB-005], it takes 2 days. Best regards, ShopLite Support"

        llm = FakeLLM([hallucinated_draft, corrected_draft])
        state = _make_state(llm_calls_used=1)

        result = resolver_node(state, llm=llm, retriever=mock_retriever)

        assert result["citations"] == ["KB-005"]
        assert "[KB-005]" in result["draft_body"]
        assert result["llm_calls_used"] == 3  # 1 initial + 1 hallucinated + 1 retry
        assert len(llm.calls) == 2
        # Check correction message was fed back
        assert "KB-999" in llm.calls[1][-1]["content"]

    def test_resolver_budget_exceeded_aborts(self) -> None:
        mock_retriever = MagicMock()
        llm = FakeLLM(["Should not be called"])
        state = _make_state(llm_calls_used=6, max_llm_calls=6)

        result = resolver_node(state, llm=llm, retriever=mock_retriever)

        assert result["status"] == "error"
        assert "budget exceeded" in result["error"].lower()
        assert len(llm.calls) == 0
