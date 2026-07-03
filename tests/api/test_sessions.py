from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_rag_chain, get_session_store


@pytest.fixture
def client():
    app = create_app()
    mock_store = MagicMock()
    mock_store.get_history.return_value = [
        {"role": "user", "content": "Hello", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "Hi there!", "timestamp": "2026-01-01T00:00:01"},
    ]
    mock_store.list_sessions.return_value = [
        {"id": "abc123", "round_count": 3, "created_at": "2026-01-01T00:00:00"}
    ]
    mock_store.get_full_history.return_value = [
        {"role": "user", "content": "Hello", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "Hi there!", "timestamp": "2026-01-01T00:00:01"},
    ]

    app.dependency_overrides[get_session_store] = lambda: mock_store

    mock_rag = MagicMock()
    mock_rag.query.return_value = {"answer": "测试回答", "sources": [], "confidence": 0.9}
    app.dependency_overrides[get_rag_chain] = lambda: mock_rag

    return TestClient(app)


def test_list_sessions(client):
    resp = client.get("/api/v1/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["id"] == "abc123"


def test_get_session_detail(client):
    resp = client.get("/api/v1/sessions/abc123")
    assert resp.status_code == 200
    data = resp.json()
    assert "rounds" in data
    assert len(data["rounds"]) == 2


def test_delete_session(client):
    resp = client.delete("/api/v1/sessions/abc123")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True}


def test_query_with_session_id(client):
    resp = client.post("/api/v1/query", json={"query": "你好", "session_id": "test-session"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "test-session"
    assert "answer" in data


def test_query_without_session_id_auto_generates(client):
    resp = client.post("/api/v1/query", json={"query": "你好"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] is not None
    assert len(data["session_id"]) > 0
