"""
app/agents/graph.py

Responsibility: LangGraph StateGraph orchestration for the support desk.
Wires Triage -> Confidence Gate -> Resolver <-> Critic (Revision Loop) -> Action Gate (Interrupt / HITL).
Enforces the hard cap on LLM calls per ticket and persists thread checkpoints via SQLite.
"""

from __future__ import annotations

import functools
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy.orm import Session

from app.agents.nodes.critic import critic_node
from app.agents.nodes.resolver import resolver_node
from app.agents.nodes.triage import triage_node
from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.db.enums import TicketStatus
from app.db.models import ApprovalDecision, Draft, Outbox, Setting, Ticket
from app.services.policies.models import PolicySet
from app.services.providers.base import BaseLLM

if TYPE_CHECKING:
    from app.services.kb.retriever import KBRetriever

logger = get_logger(__name__)

# Backend root directory for default checkpoint persistence
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_CHECKPOINTER_INSTANCE: SqliteSaver | None = None


def get_checkpointer(db_path: Path | str | None = None) -> SqliteSaver:
    """
    Retrieve or create a persistent singleton SqliteSaver instance.
    Windows-friendly with resolved paths and check_same_thread=False.
    """
    global _CHECKPOINTER_INSTANCE
    if _CHECKPOINTER_INSTANCE is None:
        if db_path is None:
            db_path = _BACKEND_ROOT / "data" / "checkpoints.db"
        resolved_path = Path(db_path).resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(resolved_path), check_same_thread=False)
        _CHECKPOINTER_INSTANCE = SqliteSaver(conn)
    return _CHECKPOINTER_INSTANCE


def reset_checkpointer_instance() -> None:
    """Reset global checkpointer singleton (primarily for testing)."""
    global _CHECKPOINTER_INSTANCE
    _CHECKPOINTER_INSTANCE = None


# ---------------------------------------------------------------------------
# Routing Functions & Pure Business Logic
# ---------------------------------------------------------------------------


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


def route_after_resolver(state: AgentState) -> str:
    """
    Conditional routing edge following resolver node.
    - KB gap detected or draft absent -> "kb_gap" (terminates at END)
    - Draft successfully generated -> "critic"
    """
    if state.get("kb_gap") or not state.get("draft_body"):
        return "kb_gap"
    return "critic"


def route_after_critic(state: AgentState) -> str:
    """
    Conditional routing edge following critic node.
    - Verdict is 'revise' AND revisions_used < max_revision_loops -> "revise" (loops back to resolver)
    - Otherwise -> "action_gate"
    """
    verdict = state.get("critic_verdict")
    revisions_used = state.get("revisions_used", 0)
    settings = get_settings()
    max_revisions = settings.app.max_revision_loops

    if verdict == "revise" and revisions_used < max_revisions:
        logger.info(
            "Critic requested revision within budget; routing back to resolver",
            extra={
                "ticket_id": state.get("ticket_id"),
                "revisions_used": revisions_used,
                "max_revisions": max_revisions,
            },
        )
        return "revise"

    return "action_gate"


def evaluate_action_gate(
    critic_verdict: str | None,
    category: str | None,
    auto_send_allowlist: list[str],
    never_auto_send: list[str],
    auto_send_enabled: bool,
) -> bool:
    """
    Pure function to evaluate auto-send eligibility against the 4 gating criteria:
    1. Critic verdict must be exactly 'pass'.
    2. Category must be present and in taxonomy auto_send_allowlist.
    3. Category must NOT be in taxonomy never_auto_send.
    4. auto_send_enabled master switch must be True.
    """
    if critic_verdict != "pass":
        return False
    if not category:
        return False
    if category in never_auto_send:
        return False
    if category not in auto_send_allowlist:
        return False
    if not auto_send_enabled:
        return False
    return True


def is_auto_send_enabled(db: Session | None = None) -> bool:
    """
    Check if auto-send is enabled.
    Priority: DB Setting table > configs/app.yaml.
    """
    if db is not None:
        setting = db.query(Setting).filter_by(key="auto_send_enabled").first()
        if setting is not None:
            return setting.value.lower() in ("true", "1", "yes")
    return get_settings().app.auto_send_enabled


def set_auto_send_enabled(enabled: bool, db: Session) -> None:
    """Persist auto_send_enabled state in the database."""
    setting = db.query(Setting).filter_by(key="auto_send_enabled").first()
    if not setting:
        setting = Setting(
            key="auto_send_enabled",
            value=str(enabled).lower(),
            description="Master auto-send switch",
        )
        db.add(setting)
    else:
        setting.value = str(enabled).lower()
    db.commit()


# ---------------------------------------------------------------------------
# Node Definitions
# ---------------------------------------------------------------------------


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


def action_gate_node(state: AgentState) -> dict[str, Any]:
    """
    Action Gate node:
    - If auto-send criteria pass: status -> auto_sent, sent_by -> auto.
    - Otherwise: pauses execution via LangGraph interrupt, persisting state to checkpoint.
      When resumed with a human decision (approve, edit, reject), completes the transition.
    """
    ticket_id = state.get("ticket_id", "unknown")
    settings = get_settings()

    auto_send_enabled = state.get("auto_send_enabled")
    if auto_send_enabled is None:
        auto_send_enabled = settings.app.auto_send_enabled

    can_auto_send = evaluate_action_gate(
        critic_verdict=state.get("critic_verdict"),
        category=state.get("category"),
        auto_send_allowlist=settings.taxonomy.auto_send_allowlist,
        never_auto_send=settings.taxonomy.never_auto_send,
        auto_send_enabled=bool(auto_send_enabled),
    )

    if can_auto_send:
        logger.info(
            "Action gate approved auto-send",
            extra={"ticket_id": ticket_id, "category": state.get("category")},
        )
        return {
            "status": TicketStatus.AUTO_SENT.value,
            "sent_by": "auto",
            "final_body": state.get("draft_body"),
        }

    # Needs human approval -> interrupt execution!
    logger.info(
        "Action gate halting for human approval via interrupt",
        extra={
            "ticket_id": ticket_id,
            "category": state.get("category"),
            "critic_verdict": state.get("critic_verdict"),
        },
    )

    human_response = interrupt(
        {
            "ticket_id": ticket_id,
            "draft_body": state.get("draft_body"),
            "critic_verdict": state.get("critic_verdict"),
            "critic_violations": state.get("critic_violations", []),
        }
    )

    # Resumed via Command(resume={...})
    if not isinstance(human_response, dict):
        human_response = {"action": "approve"}

    action = human_response.get("action", "approve")
    if action in ("approve", "edit"):
        final_body = (
            human_response.get("edited_body")
            if action == "edit"
            else state.get("draft_body")
        )
        return {
            "status": TicketStatus.SENT.value,
            "sent_by": "human",
            "final_body": final_body,
            "human_decision": human_response,
        }
    else:  # reject
        return {
            "status": TicketStatus.REJECTED.value,
            "sent_by": None,
            "final_body": None,
            "human_decision": human_response,
        }


# ---------------------------------------------------------------------------
# Graph Compilation Factory
# ---------------------------------------------------------------------------


def create_agent_graph(
    llm: BaseLLM | None = None,
    retriever: KBRetriever | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    policy_set: PolicySet | None = None,
) -> Any:
    """
    Build and compile the Phase 4 LangGraph StateGraph.
    Wires Triage -> Confidence Gate -> Resolver <-> Critic -> Action Gate (Interrupt).
    """
    _triage = functools.partial(triage_node, llm=llm) if llm else triage_node
    _resolver = (
        functools.partial(resolver_node, llm=llm, retriever=retriever)
        if (llm or retriever)
        else resolver_node
    )
    _critic = (
        functools.partial(critic_node, llm=llm, policy_set=policy_set)
        if (llm or policy_set)
        else critic_node
    )

    if checkpointer is None:
        checkpointer = get_checkpointer()

    workflow = StateGraph(AgentState)

    # 1. Register Nodes
    workflow.add_node("triage", _triage)
    workflow.add_node("confidence_abstain", confidence_abstain_node)
    workflow.add_node("resolver", _resolver)
    workflow.add_node("critic", _critic)
    workflow.add_node("action_gate", action_gate_node)

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

    workflow.add_conditional_edges(
        "resolver",
        route_after_resolver,
        {
            "kb_gap": END,
            "critic": "critic",
        },
    )

    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "revise": "resolver",
            "action_gate": "action_gate",
        },
    )

    workflow.add_edge("action_gate", END)

    return workflow.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Graph Execution & Lifecycle Runners
# ---------------------------------------------------------------------------


def run_ticket_graph(
    ticket_id: str,
    db: Session,
    llm: BaseLLM | None = None,
    retriever: KBRetriever | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    policy_set: PolicySet | None = None,
    auto_send_override: bool | None = None,
) -> AgentState:
    """
    Execute the agent graph synchronously for a specific ticket.
    Runs until either graph completion (e.g. auto_sent, untriaged, kb_gap)
    or graph interrupt (action_gate pausing for human approval).
    Persists intermediate drafts and ticket updates to the database.
    """
    ticket = db.query(Ticket).filter_by(id=ticket_id).first()
    if not ticket:
        raise NotFoundError(f"Ticket '{ticket_id}' not found in database")

    ticket.status = TicketStatus.TRIAGING.value
    db.commit()

    settings = get_settings()
    active_auto_send = (
        auto_send_override
        if auto_send_override is not None
        else is_auto_send_enabled(db)
    )

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
        "triage_reasoning": None,
        "retrieved_chunks": [],
        "kb_gap": False,
        "draft_body": None,
        "citations": [],
        "draft_history": [],
        "critic_verdict": None,
        "critic_violations": [],
        "critic_reasoning": None,
        "sent_by": None,
        "final_body": None,
        "human_decision": None,
        "revisions_used": 0,
        "llm_calls_used": 0,
        "max_llm_calls": settings.app.max_llm_calls_per_ticket,
        "error": None,
        "auto_send_enabled": active_auto_send,
    }

    if checkpointer is None:
        checkpointer = get_checkpointer()

    graph = create_agent_graph(
        llm=llm,
        retriever=retriever,
        checkpointer=checkpointer,
        policy_set=policy_set,
    )
    config = {"configurable": {"thread_id": ticket.id}}

    logger.info("Executing agent graph for ticket", extra={"ticket_id": ticket.id})
    result_state: AgentState = graph.invoke(initial_state, config=config)

    # Check whether execution paused on an interrupt
    state_snapshot = graph.get_state(config)
    is_interrupted = bool(state_snapshot.next)

    # Synchronize drafts to database
    draft_history = result_state.get("draft_history", [])
    if draft_history:
        for entry in draft_history:
            existing = (
                db.query(Draft)
                .filter_by(ticket_id=ticket.id, version=entry["version"])
                .first()
            )
            if not existing:
                draft = Draft(
                    ticket_id=ticket.id,
                    version=entry["version"],
                    body=entry["body"],
                    citations=entry.get("citations", []),
                    created_by_node=entry.get("created_by_node", "resolver"),
                )
                db.add(draft)
    elif (fallback_body := result_state.get("draft_body")):
        # Fallback if draft_history was not populated
        existing = db.query(Draft).filter_by(ticket_id=ticket.id, version=1).first()
        if not existing:
            draft = Draft(
                ticket_id=ticket.id,
                version=1,
                body=fallback_body,
                citations=result_state.get("citations", []),
                created_by_node="resolver",
            )
            db.add(draft)

    # Synchronize metadata and classification
    ticket.category = result_state.get("category")
    ticket.priority = result_state.get("priority")
    ticket.sentiment = result_state.get("sentiment")
    ticket.churn_risk = result_state.get("churn_risk")
    ticket.confidence = result_state.get("confidence")
    ticket.kb_gap = bool(result_state.get("kb_gap", False))
    ticket.critic_verdict = result_state.get("critic_verdict")
    ticket.critic_violations = result_state.get("critic_violations")
    ticket.critic_reasoning = result_state.get("critic_reasoning")

    if is_interrupted:
        ticket.status = TicketStatus.PENDING_REVIEW.value
        result_state["status"] = TicketStatus.PENDING_REVIEW.value
    else:
        final_status = result_state.get("status", TicketStatus.PENDING_REVIEW.value)
        ticket.status = final_status
        if final_status == TicketStatus.AUTO_SENT.value:
            final_body = (
                result_state.get("final_body")
                or result_state.get("draft_body")
                or ""
            )
            outbox = Outbox(
                ticket_id=ticket.id,
                final_body=final_body,
                sent_by="auto",
                sent_at=datetime.now(UTC),
            )
            db.add(outbox)

    db.commit()
    db.refresh(ticket)

    logger.info(
        "Agent graph step finished for ticket",
        extra={
            "ticket_id": ticket.id,
            "final_status": ticket.status,
            "is_interrupted": is_interrupted,
            "critic_verdict": ticket.critic_verdict,
            "llm_calls_used": result_state.get("llm_calls_used", 0),
        },
    )

    return result_state


def resume_ticket_graph(
    ticket_id: str,
    decision: dict[str, Any],
    db: Session,
    checkpointer: BaseCheckpointSaver | None = None,
) -> AgentState:
    """
    Resume an interrupted ticket graph from its checkpoint with a human decision.
    Validates ticket is pending_review, passes decision to LangGraph Command(resume=decision),
    persists outbox send or rejection status, and records the ApprovalDecision row.
    """
    ticket = db.query(Ticket).filter_by(id=ticket_id).first()
    if not ticket:
        raise NotFoundError(f"Ticket '{ticket_id}' not found")
    if ticket.status != TicketStatus.PENDING_REVIEW.value:
        raise ValidationError(
            f"Ticket '{ticket_id}' is in status '{ticket.status}', expected 'pending_review'"
        )

    action = decision.get("action")
    if action not in ("approve", "edit", "reject"):
        raise ValidationError(f"Invalid decision action '{action}'")
    if action == "edit" and not decision.get("edited_body"):
        raise ValidationError("edited_body is required when action is 'edit'")
    if action == "reject" and not decision.get("reason"):
        raise ValidationError("reason is required when action is 'reject'")

    if checkpointer is None:
        checkpointer = get_checkpointer()

    graph = create_agent_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": ticket.id}}

    state_snapshot = graph.get_state(config)
    if not state_snapshot.next:
        raise ValidationError(
            f"Ticket '{ticket_id}' is not paused on an active interrupt checkpoint"
        )

    # Resume the graph at the interrupted action_gate node
    final_state: AgentState = graph.invoke(Command(resume=decision), config=config)

    latest_draft = (
        db.query(Draft)
        .filter_by(ticket_id=ticket.id)
        .order_by(Draft.version.desc())
        .first()
    )
    original_body = latest_draft.body if latest_draft else ""

    now = datetime.now(UTC)
    latency_ms: int | None = None
    if latest_draft and latest_draft.created_at:
        dt = latest_draft.created_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = now - dt
        latency_ms = max(0, int(delta.total_seconds() * 1000))

    if action in ("approve", "edit"):
        final_body = (
            decision.get("edited_body") if action == "edit" else original_body
        )
        ticket.status = TicketStatus.SENT.value
        outbox = Outbox(
            ticket_id=ticket.id,
            final_body=final_body,
            sent_by="human",
            sent_at=now,
        )
        db.add(outbox)
        logger.info(
            "Ticket sent via human approval",
            extra={"ticket_id": ticket.id, "action": action, "latency_ms": latency_ms},
        )
    else:  # reject
        ticket.status = TicketStatus.REJECTED.value
        logger.info(
            "Ticket rejected by human reviewer",
            extra={"ticket_id": ticket.id, "reason": decision.get("reason")},
        )

    # Record the audit decision
    approval_record = ApprovalDecision(
        ticket_id=ticket.id,
        decision=action,
        original_body=original_body,
        edited_body=decision.get("edited_body") if action == "edit" else None,
        reason=decision.get("reason"),
        latency_ms=latency_ms,
        decided_at=now,
    )
    db.add(approval_record)

    db.commit()
    db.refresh(ticket)

    return final_state
