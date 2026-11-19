"""
scripts/simulate_tickets.py

Responsibility: Feed realistic fixture tickets into the support triage system,
display live agent graph execution progress, and output a structured diagnostic table.

Usage:
    python scripts/simulate_tickets.py [--base-url http://127.0.0.1:8000] [--delay 0.5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure backend root is in sys.path
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import httpx

_FIXTURES_PATH = _BACKEND_ROOT / "data" / "fixtures" / "tickets_seed.json"


def load_fixtures() -> list[dict[str, Any]]:
    """Load sample ticket payloads from fixtures."""
    if not _FIXTURES_PATH.exists():
        raise FileNotFoundError(f"Fixture file not found: {_FIXTURES_PATH}")
    with open(_FIXTURES_PATH, encoding="utf-8") as f:
        return json.load(f)


def _check_live_server(base_url: str) -> bool:
    """Check if the backend server is running and responding to health check."""
    try:
        resp = httpx.get(f"{base_url}/api/health", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def run_simulation(
    base_url: str = "http://127.0.0.1:8000",
    delay_seconds: float = 0.5,
    limit: int | None = None,
    show_drafts: bool = True,
) -> None:
    """Submit tickets and poll for results, falling back to in-process TestClient if server is down."""
    fixtures = load_fixtures()
    if limit is not None:
        fixtures = fixtures[:limit]
    is_live = _check_live_server(base_url)

    print("=" * 80)
    print(f"ShopLite Support Triage Simulator — {len(fixtures)} Tickets")
    if is_live:
        print(f"Target: Live FastAPI server at {base_url}")
    else:
        print("Target: In-Process FastAPI TestClient (live server not detected)")
    print("=" * 80)

    client: Any
    if is_live:
        client = httpx.Client(base_url=base_url, timeout=30.0)
    else:
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)

    results: list[dict[str, Any]] = []

    for idx, item in enumerate(fixtures, start=1):
        subject_preview = (item["subject"][:32] + "...") if len(item["subject"]) > 32 else item["subject"]
        print(f"[{idx:02d}/{len(fixtures):02d}] Submitting: {item['customer_name']} | \"{subject_preview}\"...", end="", flush=True)

        t0 = time.time()
        # 1. POST ticket
        resp = client.post("/api/tickets", json=item)
        if resp.status_code not in (200, 202):
            print(f" [FAILED: HTTP {resp.status_code}]")
            continue

        ticket_id = resp.json()["id"]

        # 2. Poll until status transitions beyond 'new' / 'triaging'
        ticket_data: dict[str, Any] = {}
        max_wait = 30.0
        start_wait = time.time()

        while time.time() - start_wait < max_wait:
            detail_resp = client.get(f"/api/tickets/{ticket_id}")
            if detail_resp.status_code == 200:
                ticket_data = detail_resp.json()
                status = ticket_data.get("status", "")
                if status not in ("new", "triaging"):
                    break
            time.sleep(0.3)

        elapsed = time.time() - t0
        status = ticket_data.get("status", "unknown")
        category = ticket_data.get("category") or "—"
        priority = ticket_data.get("priority") or "—"
        kb_gap = ticket_data.get("kb_gap", False)

        print(f" Done ({elapsed:.1f}s) -> {status.upper()} [{category} / {priority}] (kb_gap={kb_gap})")
        if show_drafts and ticket_data.get("drafts"):
            for d in ticket_data["drafts"]:
                cits = d.get("citations", [])
                print(f"   [DRAFT] Citations: {cits}")
                body_snippet = d.get("body", "").strip()
                for line in body_snippet.split("\n"):
                    print(f"      | {line}")
        results.append(ticket_data)

        if delay_seconds > 0 and idx < len(fixtures):
            time.sleep(delay_seconds)

    if hasattr(client, "close"):
        client.close()

    # Render summary table
    print("\n" + "=" * 105)
    print(f"{'TICKET ID':<16} | {'CUSTOMER':<16} | {'CATEGORY':<18} | {'PRIO':<5} | {'CONF':<6} | {'STATUS':<15} | {'KB GAP':<7} | {'CITATIONS'}")
    print("-" * 105)

    categories_count: dict[str, int] = {}
    statuses_count: dict[str, int] = {}
    kb_gaps = 0

    for r in results:
        tid = r.get("id", "")
        cname = (r.get("customer_name", "")[:14] + "..") if len(r.get("customer_name", "")) > 16 else r.get("customer_name", "")
        cat = r.get("category") or "none"
        prio = r.get("priority") or "—"
        conf = f"{r.get('confidence'):.2f}" if r.get("confidence") is not None else "—"
        stat = r.get("status", "")
        gap = "YES" if r.get("kb_gap") else "no"
        drafts = r.get("drafts", [])
        cits = ",".join(drafts[0].get("citations", [])) if drafts else "—"

        categories_count[cat] = categories_count.get(cat, 0) + 1
        statuses_count[stat] = statuses_count.get(stat, 0) + 1
        if r.get("kb_gap"):
            kb_gaps += 1

        print(f"{tid:<16} | {cname:<16} | {cat:<18} | {prio:<5} | {conf:<6} | {stat:<15} | {gap:<7} | {cits}")

    print("=" * 105)
    print(f"Total Tickets Processed : {len(results)}")
    print(f"Statuses Breakdown      : {statuses_count}")
    print(f"Categories Breakdown    : {categories_count}")
    print(f"Knowledge Base Gaps     : {kb_gaps}")
    print("=" * 105)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate support tickets against Support Triage Agent")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI backend URL")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay in seconds between tickets")
    parser.add_argument("--limit", type=int, default=None, help="Maximum tickets to process")
    parser.add_argument("--no-drafts", dest="show_drafts", action="store_false", help="Suppress draft previews")
    parser.set_defaults(show_drafts=True)
    args = parser.parse_args()

    run_simulation(
        base_url=args.base_url,
        delay_seconds=args.delay,
        limit=args.limit,
        show_drafts=args.show_drafts,
    )
