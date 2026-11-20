"""
app/agents/state.py

Responsibility: Strongly typed state definition for the LangGraph agent graph.
State is passed between all nodes and checkpoints, tracking ticket metadata,
triage outputs, retrieval hits, drafted replies, citation lists, and LLM call budgets.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """
    Typed state dict for ticket triage and resolution graph execution.
    """

    # Ticket metadata
    ticket_id: str
    customer_name: str
    channel: str
    subject: str
    body: str

    # Lifecycle status
    status: str

    # Triage node outputs
    category: str | None
    priority: str | None
    sentiment: str | None
    churn_risk: str | None
    confidence: float | None
    triage_reasoning: str | None

    # Resolver node outputs
    retrieved_chunks: list[dict[str, Any]]
    kb_gap: bool
    draft_body: str | None
    citations: list[str]
    draft_history: list[dict[str, Any]]

    # Critic node outputs
    critic_verdict: str | None
    critic_violations: list[dict[str, Any]]
    critic_reasoning: str | None

    # Gate & Human Decision outputs
    sent_by: str | None
    final_body: str | None
    human_decision: dict[str, Any] | None
    auto_send_enabled: bool

    # Iteration & Budget counters
    revisions_used: int
    llm_calls_used: int
    max_llm_calls: int

    # Error handling & diagnostics
    error: str | None
