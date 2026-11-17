"""
tests/test_kb_chunker.py

Unit tests for article-aware chunking and frontmatter parsing.
Tests heading splits, breadcrumb preservation, overlap, and hash stability.
"""

from __future__ import annotations

import pytest

from app.core.errors import ValidationError
from app.services.kb.chunkers import ArticleAwareChunker, KBChunk
from app.services.kb.loaders import KBArticle, parse_article_text


def test_parse_valid_frontmatter() -> None:
    raw = """---
id: KB-100
title: Test Article Title
category: returns
tags: [tag1, tag2]
---

# Test Article Title

This is the body content.
"""
    article = parse_article_text(raw, source_name="test.md")
    assert article.id == "KB-100"
    assert article.title == "Test Article Title"
    assert article.category == "returns"
    assert article.tags == ["tag1", "tag2"]
    assert "This is the body content." in article.content
    assert len(article.content_hash) == 64  # SHA-256


def test_parse_missing_frontmatter_raises() -> None:
    raw = "# No Frontmatter\nJust body content."
    with pytest.raises(ValidationError, match="missing valid YAML frontmatter"):
        parse_article_text(raw)


def test_parse_missing_id_raises() -> None:
    raw = """---
title: Missing ID
category: general
---
# Content
"""
    with pytest.raises(ValidationError, match="missing mandatory 'id'"):
        parse_article_text(raw)


def test_chunker_splits_on_headings() -> None:
    raw = """---
id: KB-101
title: Heading Split Test
category: test
tags: [test]
---

# Heading Split Test

Overview paragraph describing the feature.

## Section One: Policy Details
Details for section one with some specifics.

## Section Two: Contact Methods
Contact information details here.
"""
    article = parse_article_text(raw)
    chunker = ArticleAwareChunker(max_chars=200)  # small max_chars to force separation
    chunks = chunker.chunk_article(article)

    assert len(chunks) >= 2
    # Check breadcrumb presence
    assert any("Section One: Policy Details" in c.text for c in chunks)
    assert any("Section Two: Contact Methods" in c.text for c in chunks)
    # Check IDs
    assert chunks[0].id == "KB-101_0"
    assert chunks[0].article_id == "KB-101"


def test_chunk_hash_stability() -> None:
    raw = """---
id: KB-102
title: Hash Stability
category: test
tags: [stable]
---

# Hash Stability
Deterministic text that must always produce the same hash.
"""
    article1 = parse_article_text(raw)
    article2 = parse_article_text(raw)
    chunker = ArticleAwareChunker()

    chunks1 = chunker.chunk_article(article1)
    chunks2 = chunker.chunk_article(article2)

    assert len(chunks1) == len(chunks2)
    assert chunks1[0].content_hash == chunks2[0].content_hash
    assert article1.content_hash == article2.content_hash


def test_chunker_splits_long_section_with_overlap() -> None:
    p1 = "Paragraph one with detailed explanation about return rules. " * 5
    p2 = "Paragraph two with additional shipping constraints and instructions. " * 5
    p3 = "Paragraph three with final warranty provisions and closing notes. " * 5

    raw = f"""---
id: KB-103
title: Long Section
category: test
---

# Long Section
## Detailed Policies
{p1}

{p2}

{p3}
"""
    article = parse_article_text(raw)
    chunker = ArticleAwareChunker(max_chars=400, overlap_chars=100)
    chunks = chunker.chunk_article(article)

    assert len(chunks) >= 2
    # Verify each chunk has breadcrumb header
    for chunk in chunks:
        assert "# Long Section" in chunk.text
