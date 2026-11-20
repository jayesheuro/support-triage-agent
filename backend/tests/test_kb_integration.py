"""
tests/test_kb_integration.py

Integration tests using the live Ollama embedder and LanceDB.
Marked with @pytest.mark.integration; automatically skipped if Ollama is not running.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.core.config import get_settings
from app.services.kb.pipeline import KBPipeline
from app.services.kb.retriever import KBRetriever


def _is_ollama_ready() -> bool:
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.integration
def test_e2e_ollama_ingestion_and_retrieval(tmp_path: Path) -> None:
    if not _is_ollama_ready():
        pytest.skip("Ollama is not running locally — skipping integration test.")

    settings = get_settings()
    if settings.app.provider != "ollama":
        pytest.skip("Active provider is not ollama — skipping integration test.")

    # 1. Run pipeline against the real ShopLite KB directory
    kb_dir = Path(__file__).resolve().parent.parent / "data" / "kb"
    assert kb_dir.exists(), "data/kb directory must exist"

    pipeline = KBPipeline.create_default(force_recreate=True)
    summary = pipeline.ingest_directory(kb_dir, force=True)

    assert summary.total_articles >= 20
    assert summary.indexed_articles >= 20
    assert summary.total_chunks >= 20

    # 2. Test idempotency — re-running should skip all
    summary_second = pipeline.ingest_directory(kb_dir, force=False)
    assert summary_second.skipped_articles == summary.total_articles
    assert summary_second.indexed_articles == 0

    # 3. Test retrieval against relevant query
    retriever = KBRetriever.create_default()
    hits = retriever.retrieve("refund for duplicate charge", top_k=3)

    assert len(hits) > 0
    # Relevant chunk should come from KB-002 or KB-003
    top_article_ids = [h.article_id for h in hits]
    assert "KB-003" in top_article_ids or "KB-002" in top_article_ids
    assert hits[0].score >= retriever.relevance_threshold
    assert retriever.has_kb_gap(hits) is False

    # 4. Test nonsense query
    nonsense_hits = retriever.retrieve("quantum superposition of extraterrestrial interstellar banana", top_k=3)
    # The best score for unrelated nonsense should fall below relevance threshold
    # or yield a significantly lower score
    if nonsense_hits:
        assert retriever.best_score(nonsense_hits) < 0.60
