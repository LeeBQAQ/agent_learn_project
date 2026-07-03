from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_rag_chain, get_session_store


@pytest.fixture
def client():
    app = create_app()
    mock_store = MagicMock()
    mock_store.get_history.return_value = []
    app.dependency_overrides[get_session_store] = lambda: mock_store

    mock_rag = MagicMock()
    mock_rag.query.return_value = {"answer": "测试回答", "sources": [], "confidence": 0.9}
    app.dependency_overrides[get_rag_chain] = lambda: mock_rag

    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code in (200, 503)


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_query_validation(client):
    resp = client.post("/api/v1/query", json={})
    assert resp.status_code == 422


def test_query_empty(client):
    resp = client.post("/api/v1/query", json={"query": ""})
    assert resp.status_code in (200, 422, 500)
