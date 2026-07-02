import json
from unittest.mock import MagicMock
from src.core.session_store import SessionStore
from src.core.config import RAGConfig


def test_get_history_returns_recent_rounds():
    config = RAGConfig()
    store = SessionStore(config)

    store._redis = MagicMock()
    data = json.dumps({
        "history": [
            {"role": "user", "content": "Q1", "timestamp": "2026-01-01T00:00:00"},
            {"role": "assistant", "content": "A1", "timestamp": "2026-01-01T00:00:01"},
        ],
        "created_at": "2026-01-01T00:00:00"
    })
    store._redis.get.return_value = data
    store._redis.ttl.return_value = 3600

    history = store.get_history("test-session")
    assert len(history) == 2
    assert history[0]["role"] == "user"


def test_get_history_nonexistent_session():
    config = RAGConfig()
    store = SessionStore(config)
    store._redis = MagicMock()
    store._redis.get.return_value = None

    history = store.get_history("nonexistent")
    assert history == []


def test_add_round_new_session():
    config = RAGConfig()
    store = SessionStore(config)

    store._redis = MagicMock()
    store._redis.get.return_value = None

    store.add_round("new-session", "user", "Hello")
    store._redis.setex.assert_called_once()
    call_args = store._redis.setex.call_args
    assert call_args[0][0] == "session:new-session"
    assert call_args[0][1] == 86400  # 24h in seconds


def test_add_round_trims_to_max_rounds():
    config = RAGConfig(session_max_rounds=2)
    store = SessionStore(config)

    store._redis = MagicMock()
    existing = json.dumps({
        "history": [
            {"role": "user", "content": "Q1", "timestamp": "t1"},
            {"role": "assistant", "content": "A1", "timestamp": "t2"},
        ],
        "created_at": "t1"
    })
    store._redis.get.return_value = existing

    store.add_round("session", "user", "Q2")
    saved = json.loads(store._redis.setex.call_args[0][2])
    assert len(saved["history"]) == 2
    assert saved["history"][0]["content"] == "A1"


def test_delete_session():
    config = RAGConfig()
    store = SessionStore(config)
    store._redis = MagicMock()

    store.delete_session("session-id")
    store._redis.delete.assert_called_once_with("session:session-id")


def test_list_sessions_returns_keys():
    config = RAGConfig()
    store = SessionStore(config)
    store._redis = MagicMock()
    store._redis.scan.return_value = (0, ["session:abc", "session:xyz"])
    store._redis.ttl.return_value = 3600
    store._redis.get.return_value = json.dumps({
        "history": [{"role": "user", "content": "Hi", "timestamp": "t1"}],
        "created_at": "t1"
    })

    sessions = store.list_sessions()
    assert len(sessions) == 2
    assert sessions[0]["id"] == "abc"
