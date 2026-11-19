"""
app/agents/graph.py

Responsibility: LangGraph StateGraph orchestration for the support desk.
Wires Triage -> Confidence Gate -> Resolver -> Pending Review lifecycle.
Enforces the hard cap on LLM calls per ticket and persists thread checkpoints.
"""

from __future__ import annotations

import functools
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.agents.nodes.resolver import resolver_node
from app.agents.nodes.triage import triage_node
from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.enums import TicketStatus
from app.db.models import Draft, Ticket
from app.services.kb.retriever import KBRetriever
from app.services.providers.base import BaseLLM

logger = get_logger(__name__)


def route_after_triage(state: AgentState) -> str:
    """
    Conditional routing edge following the triage node.
    - Parse failure -> "untriaged" (terminates at END)
    - Low confidence -> "low_confidence" (routes to confidence_abstain node)
    - Grounded / high confidence -> "resolver"
    """
    if state.get("status") == TicketStatus.UNTRIAGED.value:
        return "untriaged"

    confidence = state.get("confidence")
    if confidence is None:
        confidence = 0.0

    settings = get_settings()
    threshold = settings.taxonomy.confidence_threshold

    if confidence < threshold:
        logger.info(
            "Confidence gate triggered: confidence below threshold; skipping resolver",
            extra={
                "ticket_id": state.get("ticket_id"),
                "confidence": confidence,
                "threshold": threshold,
            },
        )
        return "low_confidence"

    return "resolver"


def confidence_abstain_node(state: AgentState) -> dict[str, Any]:
    """
    Node invoked when triage confidence falls below the taxonomy threshold.
    Bypasses resolver drafting and routes ticket directly to human review.
    """
    return {
        "status": TicketStatus.PENDING_REVIEW.value,
        "draft_body": None,
        "citations": [],
    }


def create_agent_graph(
    llm: BaseLLM | None = None,
    retriever: KBRetriever | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """
    Build and compile the Phase 3 LangGraph StateGraph.
    Optionally binds custom LLM, Retriever, or Checkpointer (useful for testing).
    """
    # Bind custom dependencies to node callables if provided
    _triage = functools.partial(triage_node, llm=llm) if llm else triage_node
    _resolver = (
        functools.partial(resolver_node, llm=llm, retriever=retriever)
        if (llm or retriever)
        else resolver_node
    )

    workflow = StateGraph(AgentState)

    # 1. Register Nodes
    workflow.add_node("triage", _triage)
    workflow.add_node("confidence_abstain", confidence_abstain_node)
    workflow.add_node("resolver", _resolver)

    # 2. Wire Edges
    workflow.add_edge(START, "triage")

    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "untriaged": END,
            "low_confidence": "confidence_abstain",
            "resolver": "resolver",
        },
    )

    workflow.add_edge("confidence_abstain", END)
    workflow.add_edge("resolver", END)

    return workflow.compile(checkpointer=checkpointer)


def run_ticket_graph(
    ticket_id: str,
    db: Session,
    llm: BaseLLM | None = None,
    retriever: KBRetriever | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> AgentState:
    """
    Execute the agent graph synchronously for a specific ticket.
    Loads ticket data, executes state transitions, updates database records,
    creates drafts if produced, and records the final status.
    """
    ticket = db.query(Ticket).filter_by(id=ticket_id).first()
    if not ticket:
        raise NotFoundError(f"Ticket '{ticket_id}' not found in database")

    # Mark ticket as actively triaging
    ticket.status = TicketStatus.TRIAGING.value
    db.commit()

    settings = get_settings()
    initial_state: AgentState = {
        "ticket_id": ticket.id,
        "customer_name": ticket.customer_name,
        "channel": ticket.channel,
        "subject": ticket.subject,
        "body": ticket.body,
        "status": TicketStatus.TRIAGING.value,
        "category": ticket.category,
        "priority": ticket.priority,
        "sentiment": ticket.sentiment,
        "churn_risk": ticket.churn_risk,
        "confidence": ticket.confidence,
        "retrieved_chunks": [],
        "kb_gap": False,
        "draft_body": None,
        "citations": [],
        "revisions_used": 0,
        "llm_calls_used": 0,
        "max_llm_calls": settings.app.max_llm_calls_per_ticket,
        "error": None,
    }

    graph = create_agent_graph(llm=llm, retriever=retriever, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": ticket.id}}

    logger.info("Executing agent graph for ticket", extra={"ticket_id": ticket.id})
    final_state: AgentState = graph.invoke(initial_state, config=config)

    # Sync graph results back to database
    ticket.status = final_state.get("status", TicketStatus.PENDING_REVIEW.value)
    ticket.category = final_state.get("category")
    ticket.priority = final_state.get("priority")
    ticket.sentiment = final_state.get("sentiment")
    ticket.churn_risk = final_state.get("churn_risk")
    ticket.confidence = final_state.get("confidence")
    ticket.kb_gap = bool(final_state.get("kb_gap", False))

    draft_body = final_state.get("draft_body")
    if draft_body:
        draft = Draft(
            ticket_id=ticket.id,
            version=1,
            body=draft_body,
            citations=final_state.get("citations", []),
            created_by_node="resolver",
        )
        db.add(draft)

    db.commit()
    db.refresh(ticket)

    logger.info(
        "Agent graph execution finished for ticket",
        extra={
            "ticket_id": ticket.id,
            "final_status": ticket.status,
            "category": ticket.category,
            "priority": ticket.priority,
            "kb_gap": ticket.kb_gap,
            "has_draft": bool(draft_body),
            "llm_calls_used": final_state.get("llm_calls_used", 0),
        },
    )

    return final_state
