"""
tests/test_kb_retriever.py

Unit tests for KBRetriever and relevance thresholding with FakeEmbedder.
Verifies score extraction, ranking, and the has_kb_gap abstention signal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.kb.chunkers import ArticleAwareChunker
from app.services.kb.loaders import parse_article_text
from app.services.kb.pipeline import KBPipeline
from app.services.kb.retriever import KBRetriever, RetrievedChunk
from app.services.kb.vector_store import KBVectorStore
from tests.fakes import FakeEmbedder


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dimension=16)


@pytest.fixture
def populated_retriever(tmp_path: Path, fake_embedder: FakeEmbedder) -> KBRetriever:
    store = KBVectorStore(db_path=tmp_path / "lancedb")
    pipeline = KBPipeline(
        vector_store=store,
        embedder=fake_embedder,
        chunker=ArticleAwareChunker(),
        model_name="fake-model",
    )

    art1 = parse_article_text("""---
id: KB-001
title: Returns Policy
category: returns
---
# Returns Policy
Returns are accepted within 30 days of purchase.
""")

    art2 = parse_article_text("""---
id: KB-002
title: Warranty Coverage
category: warranty
---
# Warranty Coverage
Our one-year limited warranty covers manufacturer hardware defects.
""")

    pipeline.ingest_articles([art1, art2])
    return KBRetriever(
        vector_store=store,
        embedder=fake_embedder,
        relevance_threshold=0.35,
    )


def test_retriever_returns_ranked_chunks(populated_retriever: KBRetriever) -> None:
    chunks = populated_retriever.retrieve("Returns are accepted within 30 days of purchase.", top_k=2)
    assert len(chunks) == 2
    assert isinstance(chunks[0], RetrievedChunk)
    # The chunk matching the exact query should have the highest cosine similarity
    assert chunks[0].score >= chunks[1].score


def test_retriever_empty_query_returns_empty(populated_retriever: KBRetriever) -> None:
    chunks = populated_retriever.retrieve("   ")
    assert chunks == []
    assert populated_retriever.has_kb_gap(chunks) is True


def test_has_kb_gap_logic(populated_retriever: KBRetriever) -> None:
    # High-relevance chunk
    chunks_high = [
        RetrievedChunk(
            id="1", article_id="KB-001", title="T", category="C",
            chunk_index=0, text="txt", score=0.85, content_hash="h1"
        )
    ]
    assert populated_retriever.has_kb_gap(chunks_high) is False
    assert populated_retriever.best_score(chunks_high) == 0.85

    # Low-relevance chunk (below 0.35)
    chunks_low = [
        RetrievedChunk(
            id="2", article_id="KB-001", title="T", category="C",
            chunk_index=0, text="txt", score=0.20, content_hash="h2"
        )
    ]
    assert populated_retriever.has_kb_gap(chunks_low) is True
    assert populated_retriever.best_score(chunks_low) == 0.20
