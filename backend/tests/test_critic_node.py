"""
tests/test_critic_node.py

Responsibility: Unit tests for the Critic agent node, severity-to-verdict mapping,
action gate truth table, and structured output retry with hallucinated policy IDs.
"""

import json

import pytest

from app.agents.graph import evaluate_action_gate
from app.agents.nodes.critic import (
    CriticViolation,
    critic_node,
    determine_verdict,
)
from app.agents.state import AgentState
from app.services.policies.models import PolicyRule, PolicySet
from tests.fakes import FakeLLM


@pytest.fixture
def sample_policy_set() -> PolicySet:
    return PolicySet(
        version="test_v1",
        changelog="Testing policy set",
        rules=[
            PolicyRule(
                id="POL-001",
                title="Refund Limit",
                description="Refunds over $20 require human approval",
                severity="critical",
                examples_violating=["I refunded $50"],
                examples_compliant=["Routed to human review"],
            ),
            PolicyRule(
                id="POL-004",
                title="No Discounts",
                description="No unapproved discounts",
                severity="major",
                examples_violating=["Here is 20% off"],
                examples_compliant=["Check our sales page"],
            ),
            PolicyRule(
                id="POL-010",
                title="Formatting",
                description="Clear formatting required",
                severity="minor",
                examples_violating=["wall of text"],
                examples_compliant=["bulleted text"],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 1. Pure Function: Critic Verdict Mapping
# ---------------------------------------------------------------------------


def test_determine_verdict_no_violations():
    assert determine_verdict([]) == "pass"


def test_determine_verdict_minor_only():
    violations = [
        CriticViolation(
            policy_id="POL-010",
            severity="minor",
            quoted_span="unformatted text",
            reason="Lacks structure",
        )
    ]
    assert determine_verdict(violations) == "pass"


def test_determine_verdict_major_with_revisions_available():
    violations = [
        CriticViolation(
            policy_id="POL-004",
            severity="major",
            quoted_span="Here is 20% off",
            reason="Unauthorized discount",
        )
    ]
    # revisions_used=0 < max_revision_loops=1 -> revise
    assert determine_verdict(violations, revisions_used=0, max_revision_loops=1) == "revise"


def test_determine_verdict_major_with_revisions_exhausted():
    violations = [
        CriticViolation(
            policy_id="POL-004",
            severity="major",
            quoted_span="Here is 20% off",
            reason="Unauthorized discount",
        )
    ]
    # revisions_used=1 >= max_revision_loops=1 -> escalate
    assert determine_verdict(violations, revisions_used=1, max_revision_loops=1) == "escalate"


def test_determine_verdict_critical_always_escalates():
    violations = [
        CriticViolation(
            policy_id="POL-001",
            severity="critical",
            quoted_span="Refunded $100",
            reason="Exceeds $20 limit",
        ),
        CriticViolation(
            policy_id="POL-010",
            severity="minor",
            quoted_span="text",
            reason="formatting",
        ),
    ]
    # Even if revisions_used=0, critical always escalates
    assert determine_verdict(violations, revisions_used=0, max_revision_loops=1) == "escalate"


# ---------------------------------------------------------------------------
# 2. Pure Function: Action Gate Truth Table
# ---------------------------------------------------------------------------


def test_action_gate_truth_table():
    allowlist = ["order_status", "product_question"]
    never_send = ["refund_billing", "account_access"]

    # Must satisfy all 4 criteria to auto-send:
    # 1. verdict == pass
    # 2. category in allowlist
    # 3. category NOT in never_send
    # 4. auto_send_enabled == True

    # Baseline passing case
    assert evaluate_action_gate("pass", "order_status", allowlist, never_send, True) is True
    assert evaluate_action_gate("pass", "product_question", allowlist, never_send, True) is True

    # 1. Non-pass verdicts
    assert evaluate_action_gate("revise", "order_status", allowlist, never_send, True) is False
    assert evaluate_action_gate("escalate", "order_status", allowlist, never_send, True) is False
    assert evaluate_action_gate(None, "order_status", allowlist, never_send, True) is False

    # 2. Category not in allowlist
    assert evaluate_action_gate("pass", "bug_report", allowlist, never_send, True) is False
    assert evaluate_action_gate("pass", "feedback_other", allowlist, never_send, True) is False
    assert evaluate_action_gate("pass", None, allowlist, never_send, True) is False

    # 3. Category in never_auto_send (even if someone put it in allowlist)
    assert evaluate_action_gate("pass", "refund_billing", ["refund_billing"], never_send, True) is False
    assert evaluate_action_gate("pass", "account_access", ["account_access"], never_send, True) is False

    # 4. auto_send_enabled is False
    assert evaluate_action_gate("pass", "order_status", allowlist, never_send, False) is False
    assert evaluate_action_gate("pass", "product_question", allowlist, never_send, False) is False


# ---------------------------------------------------------------------------
# 3. Critic Node with FakeLLM
# ---------------------------------------------------------------------------


def test_critic_node_pass(sample_policy_set):
    llm_output = json.dumps({
        "verdict": "pass",
        "violations": [],
        "overall_reasoning": "Draft strictly adheres to store guidelines.",
    })
    llm = FakeLLM([llm_output])

    state: AgentState = {
        "ticket_id": "T-01",
        "draft_body": "Thank you for contacting ShopLite. Per [KB-001], returns are accepted within 30 days.",
        "customer_name": "Alice",
        "category": "order_status",
        "revisions_used": 0,
        "llm_calls_used": 1,
        "max_llm_calls": 6,
    }

    result = critic_node(state=state, llm=llm, policy_set=sample_policy_set)
    assert result["critic_verdict"] == "pass"
    assert result["critic_violations"] == []
    assert result["llm_calls_used"] == 2


def test_critic_node_revise_on_major_violation(sample_policy_set):
    llm_output = json.dumps({
        "verdict": "revise",
        "violations": [
            {
                "policy_id": "POL-004",
                "severity": "major",
                "quoted_span": "Take 20% off",
                "reason": "Promised unauthorized discount.",
            }
        ],
        "overall_reasoning": "Discount code must be removed.",
    })
    llm = FakeLLM([llm_output])

    state: AgentState = {
        "ticket_id": "T-02",
        "draft_body": "Take 20% off your next purchase using code SAVE20.",
        "customer_name": "Bob",
        "category": "order_status",
        "revisions_used": 0,
        "llm_calls_used": 1,
        "max_llm_calls": 6,
    }

    result = critic_node(state=state, llm=llm, policy_set=sample_policy_set)
    assert result["critic_verdict"] == "revise"
    assert len(result["critic_violations"]) == 1
    assert result["critic_violations"][0]["policy_id"] == "POL-004"


def test_critic_node_escalate_on_critical_violation(sample_policy_set):
    llm_output = json.dumps({
        "verdict": "escalate",
        "violations": [
            {
                "policy_id": "POL-001",
                "severity": "critical",
                "quoted_span": "I refunded $75 to your card",
                "reason": "Exceeds $20 auto-limit without authorization.",
            }
        ],
        "overall_reasoning": "Critical refund threshold breached.",
    })
    llm = FakeLLM([llm_output])

    state: AgentState = {
        "ticket_id": "T-03",
        "draft_body": "I refunded $75 to your card immediately.",
        "revisions_used": 0,
        "llm_calls_used": 1,
        "max_llm_calls": 6,
    }

    result = critic_node(state=state, llm=llm, policy_set=sample_policy_set)
    assert result["critic_verdict"] == "escalate"
    assert len(result["critic_violations"]) == 1
    assert result["critic_violations"][0]["severity"] == "critical"


def test_critic_node_hallucinated_policy_id_retry(sample_policy_set):
    # First response contains a hallucinated policy id POL-999
    bad_llm_output = json.dumps({
        "verdict": "revise",
        "violations": [
            {
                "policy_id": "POL-999",
                "severity": "major",
                "quoted_span": "bad span",
                "reason": "hallucinated policy ID",
            }
        ],
        "overall_reasoning": "Testing hallucination.",
    })

    # Second response contains valid policy id POL-004
    good_llm_output = json.dumps({
        "verdict": "revise",
        "violations": [
            {
                "policy_id": "POL-004",
                "severity": "major",
                "quoted_span": "bad span",
                "reason": "corrected policy ID",
            }
        ],
        "overall_reasoning": "Corrected.",
    })

    llm = FakeLLM([bad_llm_output, good_llm_output])

    state: AgentState = {
        "ticket_id": "T-04",
        "draft_body": "Sample draft text",
        "revisions_used": 0,
        "llm_calls_used": 1,
        "max_llm_calls": 6,
    }

    result = critic_node(state=state, llm=llm, policy_set=sample_policy_set)
    assert result["critic_verdict"] == "revise"
    assert result["critic_violations"][0]["policy_id"] == "POL-004"
    # Verify that it took 2 LLM calls to recover
    assert result["llm_calls_used"] == 3


def test_critic_node_skipped_when_no_draft(sample_policy_set):
    llm = FakeLLM([])
    state: AgentState = {
        "ticket_id": "T-05",
        "draft_body": None,
        "revisions_used": 0,
        "llm_calls_used": 1,
        "max_llm_calls": 6,
    }

    result = critic_node(state=state, llm=llm, policy_set=sample_policy_set)
    assert result["critic_verdict"] == "escalate"
    assert "no draft" in result["critic_reasoning"].lower()
    # LLM was never called
    assert len(llm.calls) == 0


def test_critic_node_budget_exhaustion(sample_policy_set):
    llm = FakeLLM([])
    state: AgentState = {
        "ticket_id": "T-06",
        "draft_body": "Some draft",
        "llm_calls_used": 6,
        "max_llm_calls": 6,
    }

    result = critic_node(state=state, llm=llm, policy_set=sample_policy_set)
    assert result["status"] == "error"
    assert "budget exceeded" in result["error"].lower()
    assert len(llm.calls) == 0
