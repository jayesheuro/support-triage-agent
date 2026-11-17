"""
scripts/seed_kb.py

Responsibility: CLI seed script to ingest data/kb/ markdown articles into LanceDB.
Prints a formatted summary table displaying article counts, chunk totals, skipped
idempotency counts, and elapsed duration.

Usage:
    python scripts/seed_kb.py
    python scripts/seed_kb.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.kb.pipeline import KBPipeline

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed ShopLite Knowledge Base into LanceDB")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing of all articles, ignoring existing content hashes",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(level="INFO")

    kb_dir = BACKEND_DIR / "data" / "kb"
    if not kb_dir.exists():
        print(f"Error: Knowledge Base directory '{kb_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  ShopLite Knowledge Base Ingestion (LanceDB)")
    print("=" * 65)
    print(f"  Active Provider:    {settings.app.provider}")
    print(f"  Embedding Model:    {settings.provider.embedder.model}")
    print(f"  Vector Dimension:   {settings.provider.embedder.dimension}")
    print(f"  Articles Directory: {kb_dir}")
    print(f"  Force Re-index:     {args.force}")
    print("-" * 65)

    pipeline = KBPipeline.create_default(force_recreate=args.force)
    summary = pipeline.ingest_directory(kb_dir, force=args.force)

    print("\n" + "=" * 65)
    print("  Ingestion Summary")
    print("=" * 65)
    print(f"  Total Articles Evaluated:  {summary.total_articles}")
    print(f"  New/Updated Indexed:       {summary.indexed_articles}")
    print(f"  Unchanged Skipped:         {summary.skipped_articles}")
    print(f"  Total Chunks in DB:        {summary.total_chunks}")
    print(f"  Execution Time:            {summary.duration_sec:.2f} s")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
