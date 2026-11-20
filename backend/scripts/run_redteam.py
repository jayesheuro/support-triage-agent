"""
scripts/run_redteam.py

Responsibility: Run red-team fixture drafts against the Critic agent to evaluate
detection accuracy and policy guardrail effectiveness.
Prints caught vs missed violations per draft and summary statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add backend directory to path if running directly
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.agents.nodes.critic import critic_node  # noqa: E402
from app.agents.state import AgentState  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.services.policies.registry import get_active_policy_set  # noqa: E402
from app.services.providers.factory import get_providers  # noqa: E402


def run_redteam(provider_override: str | None = None) -> None:
    fixtures_path = _BACKEND_DIR / "data" / "fixtures" / "redteam_drafts.json"
    if not fixtures_path.exists():
        print(f"Error: Fixtures file not found at {fixtures_path}")
        sys.exit(1)

    with fixtures_path.open("r", encoding="utf-8") as f:
        fixtures = json.load(f)

    policy_set = get_active_policy_set()
    settings = get_settings()

    print("=" * 72)
    print("  SHOPLITE CRITIC RED-TEAM EVALUATION RUNNER")
    print("=" * 72)
    print(f"  Active Policy Set: {policy_set.version} ({len(policy_set.rules)} rules)")
    print(f"  Active Provider:   {provider_override or settings.app.provider}")
    print(f"  Test Fixtures:     {len(fixtures)} adversarial drafts")
    print("=" * 72 + "\n")

    try:
        _, llm, _ = get_providers()
    except Exception as e:
        print(f"Failed to initialize provider: {e}")
        sys.exit(1)

    total_expected_violations = 0
    total_caught_violations = 0
    passed_draft_count = 0  # Drafts where at least one violation was caught

    for i, fixture in enumerate(fixtures, start=1):
        f_id = fixture["id"]
        f_name = fixture["name"]
        expected_ids = set(fixture.get("expected_violations", []))
        total_expected_violations += len(expected_ids)

        state: AgentState = {
            "ticket_id": f_id,
            "customer_name": fixture.get("customer_name", "Customer"),
            "channel": fixture.get("channel", "web"),
            "category": fixture.get("category", "general"),
            "priority": fixture.get("priority", "P2"),
            "sentiment": fixture.get("sentiment", "neutral"),
            "churn_risk": fixture.get("churn_risk", "low"),
            "subject": fixture.get("ticket_subject", ""),
            "body": fixture.get("ticket_body", ""),
            "draft_body": fixture.get("draft_body", ""),
            "revisions_used": 0,
            "llm_calls_used": 0,
            "max_llm_calls": 6,
        }

        print(f"[{i:02d}/{len(fixtures):02d}] Testing {f_id}: {f_name}")
        print(f"      Expected Violations: {sorted(list(expected_ids))}")

        try:
            result = critic_node(state=state, llm=llm, policy_set=policy_set)
            caught_violations = result.get("critic_violations", [])
            verdict = result.get("critic_verdict", "unknown")
            caught_ids = {v["policy_id"] for v in caught_violations}

            matched = expected_ids.intersection(caught_ids)
            missed = expected_ids - caught_ids
            extra = caught_ids - expected_ids

            total_caught_violations += len(matched)
            if matched:
                passed_draft_count += 1

            status_indicator = "PASSED" if not missed else ("PARTIAL" if matched else "FAILED")
            print(f"      Status:   [{status_indicator}]")
            print(f"      Verdict:  {verdict.upper()} (expected: {fixture.get('expected_verdict', 'revise/escalate').upper()})")
            print(f"      Caught:   {sorted(list(matched))}")
            if missed:
                print(f"      Missed:   {sorted(list(missed))}")
            if extra:
                print(f"      Additional: {sorted(list(extra))}")
            print(f"      Reasoning: {result.get('critic_reasoning')}\n")

        except Exception as exc:
            print(f"      ERROR during evaluation: {exc}\n")

    print("=" * 72)
    print("  EVALUATION SUMMARY")
    print("=" * 72)
    catch_rate = (
        (total_caught_violations / total_expected_violations) * 100
        if total_expected_violations > 0
        else 0.0
    )
    draft_catch_rate = (
        (passed_draft_count / len(fixtures)) * 100 if fixtures else 0.0
    )

    print(f"  Drafts Flagged:     {passed_draft_count}/{len(fixtures)} ({draft_catch_rate:.1f}%)")
    print(f"  Violations Caught:  {total_caught_violations}/{total_expected_violations} ({catch_rate:.1f}%)")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run red-team drafts against Critic Agent")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider override")
    args = parser.parse_args()
    run_redteam(provider_override=args.provider)
