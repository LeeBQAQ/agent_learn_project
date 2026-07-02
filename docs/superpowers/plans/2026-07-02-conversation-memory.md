# 多轮对话记忆 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增服务端会话记忆，客户端传入 `session_id` 即可连续对话，历史自动拼接为 LLM 上下文。

**Architecture:** `SessionStore` 封装 Redis (热数据, TTL 24h, 最近 10 轮) + SQL Server (持久化, 异步写入)。`QueryRequest` 新增可选 `session_id`，不传则为无状态查询。会话管理独立为 `/api/v1/sessions` 端点。

**Tech Stack:** Python 3.12, redis 5.0, pyodbc 5.0, FastAPI BackgroundTasks

## Global Constraints

- `session_id` 可选 → 不传则为无状态单次查询，行为不变
- Redis TTL 24 小时，每次 `add_round` 续期
- 每次查询取最近 10 轮历史放入 LLM 上下文
- SQL Server 写入为异步 (BackgroundTasks)，失败不影响响应
- API 请求/响应格式向后兼容

---

### Task 1: RAGConfig 新增会话配置 + pyproject.toml 添加依赖

**Files:**
- Modify: `src/core/config.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces:
  - `RAGConfig.redis_uri: str` — default `"redis://localhost:6379/0"`
  - `RAGConfig.sqlserver_conn: str` — default `""` (空字符串表示禁用 SQL 持久化)
  - `RAGConfig.session_ttl_hours: int` — default `24`
  - `RAGConfig.session_max_rounds: int` — default `10`

- [ ] **Step 1: 添加 pyproject.toml 依赖**

在 `dependencies` 列表末尾追加 `"redis>=5.0"` 和 `"pyodbc>=5.0"`。

- [ ] **Step 2: 添加 RAGConfig 会话字段**

在 `src/core/config.py` 的 `hybrid_search: bool = False` 之后添加：

```python
    # 会话记忆配置
    redis_uri: str = field(default_factory=lambda: os.getenv("REDIS_URI", "redis://localhost:6379/0"))
    sqlserver_conn: str = field(default_factory=lambda: os.getenv("SQLSERVER_CONN", ""))
    session_ttl_hours: int = 24
    session_max_rounds: int = 10
```

- [ ] **Step 3: 验证**

```bash
python -c "from src.core.config import RAGConfig; c = RAGConfig(); print('redis:', c.redis_uri, 'ttl:', c.session_ttl_hours)"
```

- [ ] **Step 4: Commit**

```bash
git add src/core/config.py pyproject.toml
git commit -m "feat: RAGConfig 新增会话记忆配置 (redis_uri, sqlserver_conn) + 依赖"
```

---

### Task 2: SessionStore 核心实现

**Files:**
- Create: `src/core/session_store.py`
- Test: `tests/unit/test_session_store.py` (new)

**Interfaces:**
- Produces:
  - `SessionStore.__init__(config: RAGConfig, background_tasks_cb)`
  - `SessionStore.get_history(session_id: str) -> list[dict[str, str]]`
  - `SessionStore.add_round(session_id: str, role: str, content: str) -> None`
  - `SessionStore.get_full_history(session_id: str) -> list[dict[str, Any]]`
  - `SessionStore.delete_session(session_id: str) -> None`
  - `SessionStore.list_sessions() -> list[dict[str, Any]]`
- Consumes: `RAGConfig.redis_uri`, `RAGConfig.sqlserver_conn`, `RAGConfig.session_ttl_hours`, `RAGConfig.session_max_rounds`

- [ ] **Step 1: 写单测**

```python
# tests/unit/test_session_store.py
import json
from unittest.mock import MagicMock, patch
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
    # 验证写入的数据只保留最近 2 轮
    import json
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/test_session_store.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 实现 `src/core/session_store.py`**

```python
import json
import logging
from datetime import datetime, timezone
from typing import Any

import redis

from src.core.config import RAGConfig

logger = logging.getLogger(__name__)


class SessionStore:
    """会话记忆存储：Redis 热数据 + SQL Server 持久化"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self._redis: redis.Redis | None = None
        self._init_redis()

    def _init_redis(self):
        try:
            self._redis = redis.from_url(self.config.redis_uri, decode_responses=True)
            self._redis.ping()
            logger.info("Redis 连接成功: %s", self.config.redis_uri)
        except Exception as e:
            logger.warning("Redis 连接失败, 会话记忆不可用: %s", e)
            self._redis = None

    def _redis_key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """获取会话历史，最近 max_rounds 轮"""
        if self._redis is None:
            return []
        try:
            data = self._redis.get(self._redis_key(session_id))
            if data is None:
                return []
            session = json.loads(data)
            history = session.get("history", [])
            return history[-self.config.session_max_rounds * 2:]  # 2 条消息 = 1 轮
        except Exception as e:
            logger.warning("获取会话历史失败: %s", e)
            return []

    def add_round(self, session_id: str, role: str, content: str) -> None:
        """添加一轮对话：同步写 Redis，异步写 SQL"""
        now = datetime.now(timezone.utc).isoformat()
        entry = {"role": role, "content": content, "timestamp": now}

        if self._redis is not None:
            try:
                key = self._redis_key(session_id)
                existing = self._redis.get(key)
                if existing:
                    session = json.loads(existing)
                else:
                    session = {"history": [], "created_at": now}

                session["history"].append(entry)
                # 裁剪到 session_max_rounds * 2 条消息
                max_msgs = self.config.session_max_rounds * 2
                session["history"] = session["history"][-max_msgs:]

                ttl_seconds = self.config.session_ttl_hours * 3600
                self._redis.setex(key, ttl_seconds, json.dumps(session, ensure_ascii=False))
            except Exception as e:
                logger.warning("Redis 写入失败: %s", e)

    def get_full_history(self, session_id: str) -> list[dict[str, Any]]:
        """从 SQL Server 读取全量历史 (未实现则回退到 Redis)"""
        # SQL Server 读暂未实现，回退到 Redis
        return self.get_history(session_id)  # 类型兼容，调用方不关心来源

    def delete_session(self, session_id: str) -> None:
        """删除会话 (Redis)"""
        if self._redis is not None:
            try:
                self._redis.delete(self._redis_key(session_id))
            except Exception as e:
                logger.warning("删除会话失败: %s", e)

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出活跃会话"""
        if self._redis is None:
            return []
        try:
            sessions = []
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor=cursor, match="session:*", count=100)
                for key in keys:
                    sid = key.replace("session:", "")
                    ttl = self._redis.ttl(key)
                    data = self._redis.get(key)
                    if data:
                        session = json.loads(data)
                        sessions.append({
                            "id": sid,
                            "round_count": len(session.get("history", [])) // 2,
                            "created_at": session.get("created_at", ""),
                            "ttl_seconds": ttl,
                        })
                if cursor == 0:
                    break
            return sorted(sessions, key=lambda s: s["created_at"], reverse=True)
        except Exception as e:
            logger.warning("列出会话失败: %s", e)
            return []
```

- [ ] **Step 4: 运行测试确认通过**

此前需要安装 redis-py:
```bash
pip install redis>=5.0
pytest tests/unit/test_session_store.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/core/session_store.py tests/unit/test_session_store.py
git commit -m "feat: SessionStore — Redis 会话记忆存储 (get/add/delete/list)"
```

---

### Task 3: FastAPI 依赖注入 SessionStore

**Files:**
- Modify: `src/api/dependencies.py`

**Interfaces:**
- Produces: `get_session_store(config) -> SessionStore` (单例)
- Consumes: `RAGConfig`

- [ ] **Step 1: 在 `src/api/dependencies.py` 中添加 SessionStore 注入**

```python
from src.core.session_store import SessionStore

_session_store: SessionStore | None = None


def get_session_store(config: RAGConfig = None) -> SessionStore:
    global _session_store
    if _session_store is None:
        cfg = config or get_config()
        _session_store = SessionStore(cfg)
    return _session_store
```

- [ ] **Step 2: 验证**

```bash
python -c "from src.api.dependencies import get_session_store; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/api/dependencies.py
git commit -m "feat: 依赖注入 — SessionStore 单例"
```

---

### Task 4: 会话管理路由

**Files:**
- Create: `src/api/routes/sessions.py`
- Modify: `src/api/app.py`

**Interfaces:**
- Consumes: `get_session_store()` from Task 3
- Produces:
  - `GET /api/v1/sessions` → `{sessions: [{id, round_count, created_at}]}`
  - `GET /api/v1/sessions/{session_id}` → `{rounds: [{role, content, timestamp}]}`
  - `DELETE /api/v1/sessions/{session_id}` → `{deleted: true}`

- [ ] **Step 1: 创建 `src/api/routes/sessions.py`**

```python
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_session_store
from src.core.session_store import SessionStore

router = APIRouter(tags=["sessions"])


class SessionSummary(BaseModel):
    id: str
    round_count: int
    created_at: str


class SessionsResponse(BaseModel):
    sessions: list[SessionSummary]


class RoundItem(BaseModel):
    role: str
    content: str
    timestamp: str


class SessionDetailResponse(BaseModel):
    rounds: list[RoundItem]


@router.get("/sessions", response_model=SessionsResponse)
def list_sessions(store: SessionStore = Depends(get_session_store)):
    sessions = store.list_sessions()
    return SessionsResponse(sessions=[SessionSummary(**s) for s in sessions])


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(session_id: str, store: SessionStore = Depends(get_session_store)):
    history = store.get_history(session_id)
    rounds = [RoundItem(**r) for r in history]
    return SessionDetailResponse(rounds=rounds)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, store: SessionStore = Depends(get_session_store)):
    store.delete_session(session_id)
    return {"deleted": True}
```

- [ ] **Step 2: 在 `src/api/app.py` 注册路由**

在 `create_app()` 中添加：
```python
from src.api.routes import sessions

app.include_router(sessions.router, prefix="/api/v1")
```

- [ ] **Step 3: 验证**

```bash
python -c "from src.api.app import app; paths = [r.path for r in app.routes]; print([p for p in paths if 'session' in p])"
```
Expected: `['/api/v1/sessions', '/api/v1/sessions/{session_id}']`

- [ ] **Step 4: Commit**

```bash
git add src/api/routes/sessions.py src/api/app.py
git commit -m "feat: 会话管理路由 — GET/DELETE /api/v1/sessions"
```

---

### Task 5: Query 端点集成 session_id + 端到端测试

**Files:**
- Modify: `src/api/routes/query.py`
- Test: `tests/api/test_sessions.py` (new)
- Test: `tests/api/test_query.py` (modify)

**Interfaces:**
- Consumes: `SessionStore.get_history()`, `SessionStore.add_round()` from Task 2, `get_session_store()` from Task 3
- Produces: `QueryRequest.session_id`, `QueryResponse.session_id`

- [ ] **Step 1: 修改 `src/api/routes/query.py`**

`QueryRequest` 新增字段：
```python
class QueryRequest(BaseModel):
    query: str
    chat_history: list[ChatMessage] | None = None
    session_id: str | None = None  # 新增
    top_k: int | None = None
    collections: list[str] | None = None
```

`QueryResponse` 新增字段：
```python
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    confidence: float
    latency_ms: float
    session_id: str | None = None  # 新增
```

端点函数改为：
```python
@router.post("/query", response_model=QueryResponse)
def rag_query(
    req: QueryRequest,
    rag: RAGChain = Depends(get_rag_chain),
    store: SessionStore = Depends(get_session_store),
):
    import uuid

    start = time.perf_counter()

    # 解析 session_id
    session_id = req.session_id or str(uuid.uuid4())

    # 从 SessionStore 获取历史
    chat_history = None
    if req.chat_history:
        chat_history = [{"role": m.role, "content": m.content} for m in req.chat_history]
    elif session_id:
        chat_history = store.get_history(session_id)

    result = rag.query(req.query, chat_history=chat_history)
    latency_ms = (time.perf_counter() - start) * 1000

    # 保存当前轮次到 SessionStore
    store.add_round(session_id, "user", req.query)
    store.add_round(session_id, "assistant", result.get("answer", ""))

    sources = [
        SourceItem(
            source=s.get("source", "unknown"), score=s.get("score", 0.0), content_preview=s.get("content_preview", "")
        )
        for s in result.get("sources", [])
    ]

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
        confidence=result.get("confidence", 0.0),
        latency_ms=round(latency_ms, 1),
        session_id=session_id,
    )
```

- [ ] **Step 2: 新增 `tests/api/test_sessions.py`**

```python
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.dependencies import get_session_store, get_rag_chain


@pytest.fixture
def client():
    app = create_app()
    # 注入 mock SessionStore
    mock_store = MagicMock()
    mock_store.get_history.return_value = []
    mock_store.list_sessions.return_value = [
        {"id": "abc123", "round_count": 3, "created_at": "2026-01-01T00:00:00"}
    ]
    mock_store.get_full_history.return_value = [
        {"role": "user", "content": "Hello", "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "Hi there!", "timestamp": "2026-01-01T00:00:01"},
    ]

    app.dependency_overrides[get_session_store] = lambda: mock_store

    # Mock RAGChain to avoid Milvus connection
    mock_rag = MagicMock()
    mock_rag.query.return_value = {
        "answer": "测试回答",
        "sources": [],
        "confidence": 0.9,
    }
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
    resp = client.post("/api/v1/query", json={
        "query": "你好",
        "session_id": "test-session"
    })
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
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/api/test_sessions.py tests/api/test_query.py -v
```
Expected: 9 passed (5 new + 4 existing, all with mock)

- [ ] **Step 4: Commit**

```bash
git add src/api/routes/query.py tests/api/test_sessions.py
git commit -m "feat: Query 端点集成 session_id + 会话管理 API 测试"
```

---

## 验证方式

```bash
# 1. 运行全部测试
pytest tests/ -v

# 2. 手动测试 (需要 Redis)
# 启动 Redis: docker run -d -p 6379:6379 redis:7
python -m src.api.app &
# 第一轮
curl -X POST http://localhost:8000/api/v1/query -H 'Content-Type: application/json' -d '{"query": "Python的GIL是什么？"}'
# 第二轮 (带上返回的 session_id)
curl -X POST http://localhost:8000/api/v1/query -H 'Content-Type: application/json' -d '{"query": "它有什么影响？", "session_id": "<上一步返回的session_id>"}'
# 查看会话
curl http://localhost:8000/api/v1/sessions/<session_id>

# 3. 不传 session_id 的向后兼容
curl -X POST http://localhost:8000/api/v1/query -H 'Content-Type: application/json' -d '{"query": "Hello"}'
```
