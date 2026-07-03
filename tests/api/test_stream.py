import json
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

    def mock_stream(query, chat_history=None):
        yield ("status", '{"phase":"retrieving","message":"..."}')
        yield ("sources", '[{"source":"a.txt","score":0.9,"content_preview":"..."}]')
        yield ("status", '{"phase":"generating","message":"..."}')
        yield ("token", '{"content":"流式"}')
        yield ("token", '{"content":"回答"}')
        yield ("done", '{"confidence":0.85}')

    mock_rag = MagicMock()
    mock_rag.stream_query = mock_stream

    app.dependency_overrides[get_rag_chain] = lambda: mock_rag
    app.dependency_overrides[get_session_store] = lambda: mock_store
    return TestClient(app)


def test_stream_normal_flow(client):
    resp = client.post("/api/v1/query/stream", json={"query": "你好"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    body = resp.text
    lines = body.strip().split("\n\n")
    events = {}
    for block in lines:
        block_lines = block.strip().split("\n")
        event_type = block_lines[0].replace("event: ", "")
        data = block_lines[1].replace("data: ", "")
        events[event_type] = json.loads(data)

    assert "status" in events
    assert "sources" in events
    assert "token" in events
    assert "done" in events


def test_stream_error_flow():
    def mock_error_stream(query, chat_history=None):
        yield ("status", '{"phase":"retrieving"}')
        yield ("error", '{"code":"RETRIEVAL_FAILED","message":"timeout"}')

    mock_rag = MagicMock()
    mock_rag.stream_query = mock_error_stream

    with TestClient(create_app()) as c:
        c.app.dependency_overrides[get_rag_chain] = lambda: mock_rag
        c.app.dependency_overrides[get_session_store] = lambda: MagicMock()
        resp = c.post("/api/v1/query/stream", json={"query": "x"})

    body = resp.text
    assert "event: error" in body
    assert "RETRIEVAL_FAILED" in body
    assert "done" not in body


def test_stream_token_format(client):
    resp = client.post("/api/v1/query/stream", json={"query": "x"})
    body = resp.text

    token_events = [line for line in body.split("\n") if line.startswith("event: token")]
    assert len(token_events) >= 1
