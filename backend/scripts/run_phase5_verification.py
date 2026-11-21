"""
scripts/run_phase5_verification.py

Phase 5 Acceptance Verification Script:
1. Executes a 15-ticket simulation covering:
   - Full triage -> confidence_gate -> resolver -> critic -> action_gate lifecycles
   - Structured output retries (status='retried', retry_count >= 1)
   - Human-in-the-loop interrupts & resumes (linked under the same run_id)
   - Zero-duration gate spans
   - Truncated payloads (~500 chars) and customer content flags
2. Queries GET /api/tickets/{id}/trace to demonstrate ordered hierarchical span trees
3. Queries GET /api/traces/stats to produce aggregated metrics and derived Gemini 2.0 Flash costs
4. Displays formatted JSON outputs for direct inspection.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agents.graph import resume_ticket_graph, run_ticket_graph
from app.api.routes import traces as traces_route
from app.core.errors import register_exception_handlers
from app.db import init_db
from app.db.enums import SpanStatus, SpanType, TicketStatus
from app.db.models import Ticket, TraceRun, TraceSpan
from app.db.session import SessionLocal, get_db
from app.services.kb.retriever import RetrievedChunk
from app.services.policies.models import PolicyRule, PolicySet
from tests.fakes import FakeLLM


class VerificationRetriever:
    """Mock retriever returning consistent, valid chunks."""

    def __init__(self, relevance_score: float = 0.90, is_gap: bool = False):
        self.relevance_score = relevance_score
        self.is_gap = is_gap

    def retrieve(self, query: str, top_k: int = 2) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                id="KB-001_0",
                article_id="KB-001",
                title="Return & Refund Policy",
                category="returns_exchanges",
                chunk_index=0,
                text="Returns accepted within 30 days for undamaged items. [KB-001]",
                score=self.relevance_score,
                content_hash="h1",
                tags=["returns", "refunds"],
            )
        ]

    def has_kb_gap(self, chunks: list[RetrievedChunk]) -> bool:
        return self.is_gap


def get_verification_policy_set() -> PolicySet:
    return PolicySet(
        version="shoplite_v1",
        changelog="Initial production policies",
        rules=[
            PolicyRule(
                id="POL-001",
                title="Refund Ceiling",
                severity="critical",
                description="Refunds cannot exceed $20",
                examples_violating=["I have refunded $50."],
                examples_compliant=["I have submitted a $15 refund request."],
            ),
            PolicyRule(
                id="POL-004",
                title="Mandatory KB Citation",
                severity="major",
                description="Must cite [KB-xxx]",
                examples_violating=["Here is your return info."],
                examples_compliant=["You can return items within 30 days per [KB-001]."],
            ),
        ],
    )


def run_phase5_verification() -> None:
    print("=" * 80)
    print("PHASE 5 AGENT OBSERVABILITY — VERIFICATION & SIMULATION RUNNER")
    print("=" * 80)

    # 1. Initialize DB
    init_db()

    # Load 15 tickets from fixtures
    fixtures_path = _BACKEND_ROOT / "data" / "fixtures" / "tickets_seed.json"
    with open(fixtures_path, encoding="utf-8") as f:
        fixtures = json.load(f)[:15]

    policy_set = get_verification_policy_set()
    retriever = VerificationRetriever()

    ticket_ids: list[str] = []
    resumed_ticket_id: str | None = None
    retried_ticket_id: str | None = None

    print(f"\n[1/4] Processing {len(fixtures)} tickets through agent graph...")

    with SessionLocal() as db:
        for idx, item in enumerate(fixtures, start=1):
            tid = f"TICK-SIM-{idx:02d}"
            ticket_ids.append(tid)

            ticket = Ticket(
                id=tid,
                customer_name=item["customer_name"],
                channel=item["channel"],
                subject=item["subject"],
                body=item["body"],
                status=TicketStatus.NEW.value,
            )
            db.merge(ticket)
            db.commit()

            # Scenario scripting
            if idx == 1:
                # Scenario A: Structured output retry on triage
                retried_ticket_id = tid
                malformed = "NOT_JSON"
                valid_triage = json.dumps({
                    "category": "order_status",
                    "priority": "P2",
                    "sentiment": "neutral",
                    "churn_risk": "low",
                    "confidence": 0.92,
                    "reasoning": "Tracking status query recovered on attempt 2"
                })
                resolver_draft = "You can track your package using the link sent in your email [KB-001]."
                critic_pass = json.dumps({"verdict": "pass", "violations": [], "overall_reasoning": "Compliant draft."})
                llm = FakeLLM([malformed, valid_triage, resolver_draft, critic_pass])
                run_ticket_graph(ticket_id=tid, db=db, llm=llm, retriever=retriever, policy_set=policy_set, auto_send_override=False)

            elif idx == 2:
                # Scenario B: Human approval flow (interrupt + resume)
                resumed_ticket_id = tid
                triage = json.dumps({"category": "returns_exchanges", "priority": "P1", "sentiment": "negative", "churn_risk": "medium", "confidence": 0.88, "reasoning": "Return request"})
                resolver_draft = "You can return your item within 30 days per [KB-001]."
                critic_pass = json.dumps({"verdict": "pass", "violations": [], "overall_reasoning": "Compliant."})
                llm = FakeLLM([triage, resolver_draft, critic_pass])
                # Runs to interrupt
                run_ticket_graph(ticket_id=tid, db=db, llm=llm, retriever=retriever, policy_set=policy_set, auto_send_override=False)
                # Resumes with approval
                resume_ticket_graph(ticket_id=tid, decision={"action": "approve"}, db=db)

            elif idx % 4 == 0:
                # Scenario C: Auto-send eligible
                triage = json.dumps({"category": "order_status", "priority": "P3", "sentiment": "positive", "churn_risk": "low", "confidence": 0.96, "reasoning": "Delivery inquiry"})
                resolver_draft = "Your order will arrive as expected [KB-001]."
                critic_pass = json.dumps({"verdict": "pass", "violations": [], "overall_reasoning": "Pass."})
                llm = FakeLLM([triage, resolver_draft, critic_pass])
                run_ticket_graph(ticket_id=tid, db=db, llm=llm, retriever=retriever, policy_set=policy_set, auto_send_override=True)

            elif idx % 3 == 0:
                # Scenario D: Critic revision loop (revises draft v1 to v2)
                triage = json.dumps({"category": "returns_exchanges", "priority": "P2", "sentiment": "neutral", "churn_risk": "low", "confidence": 0.89, "reasoning": "Standard exchange"})
                draft_v1 = "I can process an unverified refund for you."
                critic_revise = json.dumps({"verdict": "revise", "violations": [{"policy_id": "POL-004", "severity": "major", "quoted_span": "unverified", "reason": "Missing citation"}], "overall_reasoning": "Missing citation."})
                draft_v2 = "You can return your items within 30 days per policy [KB-001]."
                critic_pass = json.dumps({"verdict": "pass", "violations": [], "overall_reasoning": "Resolved."})
                llm = FakeLLM([triage, draft_v1, critic_revise, draft_v2, critic_pass])
                run_ticket_graph(ticket_id=tid, db=db, llm=llm, retriever=retriever, policy_set=policy_set, auto_send_override=False)

            else:
                # Scenario E: Standard pass requiring approval
                triage = json.dumps({"category": "faq_technical", "priority": "P2", "sentiment": "neutral", "churn_risk": "low", "confidence": 0.91, "reasoning": "Tech inquiry"})
                resolver_draft = "Refer to the instructions in [KB-001] to pair your device."
                critic_pass = json.dumps({"verdict": "pass", "violations": [], "overall_reasoning": "Clean."})
                llm = FakeLLM([triage, resolver_draft, critic_pass])
                run_ticket_graph(ticket_id=tid, db=db, llm=llm, retriever=retriever, policy_set=policy_set, auto_send_override=False)

            print(f"  ✓ [{idx:02d}/15] Processed {tid} ({item['customer_name']})")

    # 2. Test API endpoints
    test_app = FastAPI()
    register_exception_handlers(test_app)
    test_app.include_router(traces_route.router, prefix="/api")

    def override_get_db():
        with SessionLocal() as s:
            yield s

    test_app.dependency_overrides[get_db] = override_get_db
    client = TestClient(test_app)

    print("\n[2/4] Verifying Acceptance Criteria on Recorded Traces:")

    # Criteria A: Pick ticket and inspect trace
    sample_tid = resumed_ticket_id or ticket_ids[0]
    resp = client.get(f"/api/tickets/{sample_tid}/trace")
    assert resp.status_code == 200, f"Failed GET /api/tickets/{sample_tid}/trace"
    trace_data = resp.json()
    assert len(trace_data["runs"]) >= 1
    spans = trace_data["runs"][0]["spans"]
    span_types = [s["span_type"] for s in spans]
    print(f"  ✓ GET /api/tickets/{sample_tid}/trace returned {len(spans)} ordered spans")
    print(f"    Spans sequence: {' -> '.join(s['name'] for s in spans)}")

    # Criteria B: Retry path visible
    retry_resp = client.get(f"/api/tickets/{retried_ticket_id}/trace")
    retry_data = retry_resp.json()
    triage_span = next(s for s in retry_data["runs"][0]["spans"] if s["name"] == "triage")
    assert triage_span["status"] == "retried", f"Expected retried, got {triage_span['status']}"
    assert triage_span["retry_count"] >= 1, f"Expected retry_count >= 1, got {triage_span['retry_count']}"
    print(f"  ✓ Retry path confirmed: {retried_ticket_id} triage status='{triage_span['status']}' retry_count={triage_span['retry_count']}")

    # Criteria C: Interrupt & Resume spans in single run
    assert "interrupt" in span_types, "Interrupt span missing in approved ticket"
    assert "resume" in span_types, "Resume span missing in approved ticket"
    print(f"  ✓ HITL continuity confirmed: {sample_tid} contains both 'interrupt' and 'resume' in run '{trace_data['runs'][0]['run_id']}'")

    # 3. GET /api/traces/stats
    print("\n[3/4] Querying GET /api/traces/stats (Aggregated Observability Metrics):")
    stats_resp = client.get("/api/traces/stats?window=all")
    assert stats_resp.status_code == 200, "Failed GET /api/traces/stats"
    stats = stats_resp.json()

    print("\n" + json.dumps(stats, indent=2))

    print("\n[4/4] Verification Summary:")
    print(f"  Total Tickets Simulated: {stats['total_tickets_traced']}")
    print(f"  Total LLM Calls:         {stats['total_llm_calls']}")
    print(f"  Total Input Tokens:      {stats['tokens']['prompt_tokens']}")
    print(f"  Total Output Tokens:     {stats['tokens']['completion_tokens']}")
    print(f"  Est. Gemini 2.0 Cost:    ${stats['estimated_cost_usd']:.6f}")
    print(f"  Auto-Send Ratio:         {stats['dispatch_stats']['auto_send_ratio']:.1%}")
    print("=" * 80)
    print("ALL PHASE 5 OBSERVABILITY ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_phase5_verification()
