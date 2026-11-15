"""
tests/test_health.py

Integration / endpoint tests for FastAPI shell (/api/health and /docs).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint_ok(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["provider"] == "ollama"
    assert data["chat_model"] == "llama3.2:latest"
    assert data["embed_model"] == "nomic-embed-text"
    assert "version" in data


def test_docs_endpoint(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_endpoint(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/api/health" in schema["paths"]
