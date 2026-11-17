"""
tests/test_kb_api.py

API endpoint tests for /api/kb routes via FastAPI TestClient.
Tests listing, uploading, search-testing, deletion, and re-indexing.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.kb.vector_store import KBVectorStore
from tests.fakes import FakeEmbedder


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_list_articles_empty(client: TestClient) -> None:
    response = client.get("/api/kb/articles")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_upload_invalid_file_rejected(client: TestClient) -> None:
    file_content = b"Not markdown"
    response = client.post(
        "/api/kb/articles",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert response.status_code == 400
    err_body = response.json()
    err_msg = err_body.get("message") or err_body.get("detail") or ""
    assert "markdown" in err_msg.lower()


def test_upload_missing_frontmatter_rejected(client: TestClient) -> None:
    file_content = b"# Missing Frontmatter\nContent goes here."
    response = client.post(
        "/api/kb/articles",
        files={"file": ("test.md", io.BytesIO(file_content), "text/markdown")},
    )
    assert response.status_code == 400


def test_upload_and_delete_article_lifecycle(client: TestClient) -> None:
    raw_md = """---
id: KB-999
title: Temporary Test Article
category: testing
tags: [temp, test]
---

# Temporary Test Article
This article will be uploaded, tested, and deleted.
"""
    # 1. Upload article
    response = client.post(
        "/api/kb/articles",
        files={"file": ("KB-999.md", io.BytesIO(raw_md.encode("utf-8")), "text/markdown")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "indexed"
    assert data["article_id"] == "KB-999"
    assert data["chunks_created"] >= 1

    # 2. Check listing
    list_resp = client.get("/api/kb/articles")
    assert list_resp.status_code == 200
    articles = list_resp.json()
    assert any(a["id"] == "KB-999" for a in articles)

    # 3. Test search endpoint
    search_resp = client.get("/api/kb/search?q=Temporary+Test+Article")
    assert search_resp.status_code == 200
    s_data = search_resp.json()
    assert "query" in s_data
    assert "relevance_threshold" in s_data
    assert "has_kb_gap" in s_data
    assert len(s_data["results"]) > 0

    # 4. Delete article
    del_resp = client.delete("/api/kb/articles/KB-999")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # 5. Delete again should return 404
    del_resp2 = client.delete("/api/kb/articles/KB-999")
    assert del_resp2.status_code == 404


def test_reindex_endpoint(client: TestClient) -> None:
    response = client.post("/api/kb/reindex")
    assert response.status_code == 200
    data = response.json()
    assert "total_articles" in data
    assert "indexed_articles" in data
    assert "skipped_articles" in data
    assert "total_chunks" in data
    assert "duration_sec" in data
