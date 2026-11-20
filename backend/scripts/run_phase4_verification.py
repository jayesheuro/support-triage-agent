"""
scripts/run_phase4_verification.py

Direct verification runner for Phase 4 deliverables:
- Critic verdict mapping
- Action gate truth table
- Critic node (pass, revise, escalate, hallucinated policy retry, budget cap)
- Revision loop (improves draft v1 to v2)
- Action gate auto-send (order_status vs refund_billing)
- HITL interrupt and resume (approve, edit, reject)
- REST APIs (policies, approvals, outbox)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(line_buffering=True)

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import tests.test_approvals_hitl as hitl_tests
import tests.test_critic_node as critic_tests


def _call_fixture(fixture_obj: Any) -> Any:
    """Call the underlying function of a pytest fixture directly."""
    fn: Any = getattr(fixture_obj, "__wrapped__", fixture_obj)
    return fn()


def run_all() -> None:
    print("=" * 70)
    print("  RUNNING PHASE 4 DIRECT VERIFICATION SUITE")
    print("=" * 70)

    # 1. Critic node & pure functions
    print("\n[1/2] Running Critic Node & Pure Function Tests...")
    critic_tests.test_determine_verdict_no_violations()
    print("  ✓ test_determine_verdict_no_violations passed")

    critic_tests.test_determine_verdict_minor_only()
    print("  ✓ test_determine_verdict_minor_only passed")

    critic_tests.test_determine_verdict_major_with_revisions_available()
    print("  ✓ test_determine_verdict_major_with_revisions_available passed")

    critic_tests.test_determine_verdict_major_with_revisions_exhausted()
    print("  ✓ test_determine_verdict_major_with_revisions_exhausted passed")

    critic_tests.test_determine_verdict_critical_always_escalates()
    print("  ✓ test_determine_verdict_critical_always_escalates passed")

    critic_tests.test_action_gate_truth_table()
    print("  ✓ test_action_gate_truth_table passed")

    sample_policy = _call_fixture(critic_tests.sample_policy_set)
    critic_tests.test_critic_node_pass(sample_policy)
    print("  ✓ test_critic_node_pass passed")

    critic_tests.test_critic_node_revise_on_major_violation(sample_policy)
    print("  ✓ test_critic_node_revise_on_major_violation passed")

    critic_tests.test_critic_node_escalate_on_critical_violation(sample_policy)
    print("  ✓ test_critic_node_escalate_on_critical_violation passed")

    critic_tests.test_critic_node_hallucinated_policy_id_retry(sample_policy)
    print("  ✓ test_critic_node_hallucinated_policy_id_retry passed")

    critic_tests.test_critic_node_skipped_when_no_draft(sample_policy)
    print("  ✓ test_critic_node_skipped_when_no_draft passed")

    critic_tests.test_critic_node_budget_exhaustion(sample_policy)
    print("  ✓ test_critic_node_budget_exhaustion passed")

    # 2. Revision Loop & HITL Workflows
    print("\n[2/2] Running Revision Loop, Action Gate, HITL & API Tests...")

    # Set up fixtures manually for direct execution
    db_gen = _call_fixture(hitl_tests.in_memory_db)
    db = next(db_gen)
    saver = _call_fixture(hitl_tests.memory_checkpointer)
    retriever = _call_fixture(hitl_tests.fake_retriever)
    policy = _call_fixture(hitl_tests.test_policy_set)

    hitl_tests.test_revision_loop_improves_draft_to_pass(db, saver, retriever, policy)
    print("  ✓ test_revision_loop_improves_draft_to_pass passed (draft v1 and v2 verified)")

    hitl_tests.test_revision_loop_exhaustion_pends_review(db, saver, retriever, policy)
    print("  ✓ test_revision_loop_exhaustion_pends_review passed")

    hitl_tests.test_hitl_approval_flow(db, saver, retriever, policy)
    print("  ✓ test_hitl_approval_flow passed (resumed -> status sent + outbox written)")

    hitl_tests.test_hitl_edit_flow(db, saver, retriever, policy)
    print("  ✓ test_hitl_edit_flow passed (custom edited text stored in outbox)")

    hitl_tests.test_hitl_reject_flow(db, saver, retriever, policy)
    print("  ✓ test_hitl_reject_flow passed (rejection reason recorded, no outbox send)")

    hitl_tests.test_auto_send_enabled_clean_ticket(db, saver, retriever, policy)
    print("  ✓ test_auto_send_enabled_clean_ticket passed (order_status auto-sent)")

    hitl_tests.test_never_auto_send_category_always_pends_even_if_auto_send_enabled(db, saver, retriever, policy)
    print("  ✓ test_never_auto_send_category_always_pends_even_if_auto_send_enabled passed (refund_billing pends)")

    hitl_tests.test_approvals_api_endpoints(db, saver, retriever, policy)
    print("  ✓ test_approvals_api_endpoints passed (GET /approvals, POST /decision, GET /outbox, GET /policies)")

    print("\n" + "=" * 70)
    print("  ALL 20 PHASE 4 TESTS PASSED CLEANLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
