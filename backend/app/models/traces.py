"""
app/models/traces.py

Responsibility: Pydantic schemas for the Trace API contracts.
Defines response shapes for /api/tickets/{id}/trace and /api/traces/stats.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TraceSpanResponse(BaseModel):
    """Granular trace span telemetry item."""

    id: int
    span_id: str
    parent_span_id: str | None = None
    span_type: str
    name: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: float | None = None
    status: str
    retry_count: int = 0
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    policy_version: str | None = None
    decision_summary: str | None = None
    input_excerpt: str | None = None
    output_excerpt: str | None = None
    is_customer_content: bool = False
    attributes: dict[str, Any] = Field(default_factory=dict)


class TraceRunResponse(BaseModel):
    """Single end-to-end execution run for a ticket with nested child spans."""

    run_id: str
    ticket_id: str
    started_at: str | None = None
    finished_at: str | None = None
    total_duration_ms: float | None = None
    total_llm_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    final_status: str
    spans: list[TraceSpanResponse] = Field(default_factory=list)


class TicketTraceResponse(BaseModel):
    """Complete trace history for a ticket."""

    ticket_id: str
    runs: list[TraceRunResponse] = Field(default_factory=list)


class NodeDurationStats(BaseModel):
    """Average and 95th percentile latency metrics for an agent node."""

    avg_ms: float
    p95_ms: float
    count: int


class TokenSummary(BaseModel):
    """Prompt, completion, and total token expenditure."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class PricingBasis(BaseModel):
    """Cost estimation assumptions."""

    provider: str
    input_rate_per_m: float
    output_rate_per_m: float


class DispatchStats(BaseModel):
    """Distribution of automated vs human dispatched communications."""

    auto_sent: int
    human_sent: int
    pending_review: int
    rejected: int
    auto_send_ratio: float


class TraceStatsResponse(BaseModel):
    """Aggregated observability and cost telemetry dashboard metrics."""

    window: str
    total_tickets_traced: int
    total_runs: int
    total_spans: int
    total_llm_calls: int
    tokens: TokenSummary
    estimated_cost_usd: float
    pricing_basis: PricingBasis
    avg_tokens_per_ticket: float
    node_durations: dict[str, NodeDurationStats] = Field(default_factory=dict)
    retry_rates: dict[str, float] = Field(default_factory=dict)
    kb_gap_rate: float = 0.0
    verdict_distribution: dict[str, int] = Field(default_factory=dict)
    dispatch_stats: DispatchStats
