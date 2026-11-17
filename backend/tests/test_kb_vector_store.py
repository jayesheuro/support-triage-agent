"""
tests/test_kb_vector_store.py

Unit tests for KBVectorStore and KBPipeline using FakeEmbedder.
Verifies dimension consistency enforcement, idempotency, updates, and deletion.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from app.core.errors import KBDimensionMismatchError
from app.services.kb.chunkers import ArticleAwareChunker
from app.services.kb.loaders import KBArticle, parse_article_text
from app.services.kb.pipeline import KBPipeline
from app.services.kb.vector_store import KBVectorStore
from tests.fakes import FakeEmbedder


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dimension=32)


@pytest.fixture
def sample_article() -> KBArticle:
    raw = """---
id: KB-001
title: Return Rules
category: returns
tags: [return, refund]
---
# Return Rules
## Overview
Return items within 30 days for a full refund.
"""
    return parse_article_text(raw)


def test_vector_store_initialization(tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
    store = KBVectorStore(db_path=tmp_path / "lancedb")
    store.ensure_table(model_name="fake-model", dimension=fake_embedder.dimension)

    assert store.count_chunks() == 0
    assert store.count_chunks() == 0
    assert len(store.list_articles()) == 0


def test_vector_store_refuses_dimension_mismatch(tmp_path: Path) -> None:
    store = KBVectorStore(db_path=tmp_path / "lancedb")
    store.ensure_table(model_name="fake-model", dimension=32)

    # Attempting to access with different dimension must raise KBDimensionMismatchError
    with pytest.raises(KBDimensionMismatchError, match="dimension/model mismatch"):
        store.ensure_table(model_name="fake-model", dimension=64)


def test_vector_store_refuses_model_name_mismatch(tmp_path: Path) -> None:
    store = KBVectorStore(db_path=tmp_path / "lancedb")
    store.ensure_table(model_name="model-a", dimension=32)

    with pytest.raises(KBDimensionMismatchError, match="dimension/model mismatch"):
        store.ensure_table(model_name="model-b", dimension=32)


def test_pipeline_idempotent_ingestion(
    tmp_path: Path, fake_embedder: FakeEmbedder, sample_article: KBArticle
) -> None:
    store = KBVectorStore(db_path=tmp_path / "lancedb")
    pipeline = KBPipeline(
        vector_store=store,
        embedder=fake_embedder,
        chunker=ArticleAwareChunker(),
        model_name="fake-model",
    )

    # First ingestion: article should be indexed
    summary1 = pipeline.ingest_articles([sample_article])
    assert summary1.total_articles == 1
    assert summary1.indexed_articles == 1
    assert summary1.skipped_articles == 0
    assert summary1.total_chunks > 0

    # Second ingestion with same article: should be skipped!
    summary2 = pipeline.ingest_articles([sample_article])
    assert summary2.total_articles == 1
    assert summary2.indexed_articles == 0
    assert summary2.skipped_articles == 1
    assert summary2.total_chunks == summary1.total_chunks


def test_pipeline_replaces_changed_article(
    tmp_path: Path, fake_embedder: FakeEmbedder
) -> None:
    store = KBVectorStore(db_path=tmp_path / "lancedb")
    pipeline = KBPipeline(
        vector_store=store,
        embedder=fake_embedder,
        chunker=ArticleAwareChunker(),
        model_name="fake-model",
    )

    art_v1 = parse_article_text("""---
id: KB-002
title: Shipping V1
category: shipping
---
# Shipping V1
Standard shipping takes 5 days.
""")

    art_v2 = parse_article_text("""---
id: KB-002
title: Shipping V2 Updated
category: shipping
---
# Shipping V2 Updated
Standard shipping takes 3 days now.
""")

    # Index v1
    s1 = pipeline.ingest_articles([art_v1])
    assert s1.indexed_articles == 1

    # Index v2: changed content hash -> should update
    s2 = pipeline.ingest_articles([art_v2])
    assert s2.indexed_articles == 1
    assert s2.skipped_articles == 0

    articles = store.list_articles()
    assert len(articles) == 1
    assert articles[0].title == "Shipping V2 Updated"


def test_vector_store_delete_article(
    tmp_path: Path, fake_embedder: FakeEmbedder, sample_article: KBArticle
) -> None:
    store = KBVectorStore(db_path=tmp_path / "lancedb")
    pipeline = KBPipeline(
        vector_store=store,
        embedder=fake_embedder,
        model_name="fake-model",
    )

    pipeline.ingest_articles([sample_article])
    assert store.count_chunks() > 0

    deleted = store.delete_article(sample_article.id)
    assert deleted is True
    assert store.count_chunks() == 0
    assert len(store.list_articles()) == 0
