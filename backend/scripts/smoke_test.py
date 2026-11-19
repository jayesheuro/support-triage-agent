"""
scripts/smoke_test.py

Responsibility: End-to-end smoke test for the provider abstraction. Run this
from the backend/ directory to verify the configured provider can generate text
and produce embeddings.

Usage:
    cd backend
    python scripts/smoke_test.py

Exits with code 0 on success, 1 on failure.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8")

# Add backend/ to sys.path so `app.*` imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services.providers.factory import get_providers

setup_logging(level="INFO")


def main() -> int:
    settings = get_settings()
    provider_name = settings.app.provider

    print(f"\n{'='*60}")
    print(f"  Support Triage Agent -- Provider Smoke Test")
    print(f"  Provider : {provider_name}")
    print(f"  Chat     : {settings.provider.llm.model}")
    print(f"  Embed    : {settings.provider.embedder.model}")
    print(f"{'='*60}\n")

    try:
        _, llm, embedder = get_providers()
    except Exception as exc:
        print(f"[FAIL] Factory failed: {exc}")
        return 1

    # -- LLM generate test ---------------------------------------------------
    print("[1/2] Testing LLM generate...")
    messages = [
        {"role": "system", "content": "You are a terse test assistant."},
        {"role": "user", "content": "Reply with exactly the word: OK"},
    ]
    t0 = time.monotonic()
    try:
        response = llm.generate(messages)
        latency = (time.monotonic() - t0) * 1000
    except Exception as exc:
        print(f"[FAIL] LLM generate failed: {exc}")
        return 1

    print(f"   Model     : {response.model_name}")
    print(f"   Response  : {response.text!r}")
    print(f"   Tokens    : {response.prompt_tokens} prompt + {response.completion_tokens} completion = {response.total_tokens} total")
    print(f"   Latency   : {latency:.0f} ms\n")

    # -- Embedder test --------------------------------------------------------
    print("[2/2] Testing Embedder...")
    test_sentence = "Order #12345 has not arrived after 10 days."
    t0 = time.monotonic()
    try:
        vector = embedder.embed_query(test_sentence)
        latency = (time.monotonic() - t0) * 1000
    except Exception as exc:
        print(f"[FAIL] Embedder failed: {exc}")
        return 1

    print(f"   Dimension : {len(vector)} (expected: {embedder.dimension})")
    print(f"   First 5   : {[round(v, 4) for v in vector[:5]]}")
    print(f"   Latency   : {latency:.0f} ms\n")

    if len(vector) != embedder.dimension:
        print(f"[FAIL] Dimension mismatch: got {len(vector)}, expected {embedder.dimension}")
        return 1

    print(f"{'='*60}")
    print(f"  [PASS] All smoke tests passed for provider: {provider_name}")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
