# 多轮对话记忆 — 设计文档

## Context

当前系统每次查询独立，客户端需自行管理全部对话历史。新增服务端会话记忆，客户端只需传 `session_id`，服务端自动拼接上下文。

## 架构

```
Redis (热数据, TTL 24h)  ←→  SQL Server (持久化, 全量)
       │                              │
       │ TTL + 最近 10 轮              │ 异步写入全量历史
       │                              │
       └──────── SessionStore ────────┘
                    │
                    └── FastAPI 路由
```

## 改动范围

| 文件 | 改动 |
|------|------|
| `src/core/config.py` | RAGConfig 新增 `redis_uri`、`sqlserver_conn`、会话配置 |
| `src/core/session_store.py` | 新增 SessionStore：Redis CRUD + 异步 SQL write |
| `src/api/routes/query.py` | 请求新增 `session_id`，响应返回 `session_id` |
| `src/api/routes/sessions.py` | 新增：查看/删除/列出会话 |
| `src/api/dependencies.py` | 注入 SessionStore 单例 |
| `pyproject.toml` | 新增 `redis>=5.0`、`pyodbc>=5.0` |

## API 端点

```
POST   /api/v1/query              请求新增 session_id: str | None
                                   响应新增 session_id: str
GET    /api/v1/sessions           列出活跃会话 { sessions: [{id, round_count, created_at}] }
GET    /api/v1/sessions/{id}      查看会话历史 { rounds: [{role, content, timestamp}] }
DELETE /api/v1/sessions/{id}      删除会话 (Redis + SQL)
```

## SessionStore 设计

```python
class SessionStore:
    def get_history(self, session_id: str) -> list[dict]: ...   # Redis 读，最近 10 轮
    def add_round(self, session_id: str, role: str, content: str): ...  # 写 Redis + TTL 续期 + 异步 SQL
    def get_full_history(self, session_id: str) -> list[dict]: ...  # SQL Server 读全量
    def delete_session(self, session_id: str): ...  # 删 Redis key + SQL DELETE
    def list_sessions(self) -> list[dict]: ...  # Redis SCAN 活跃 session
```

## Redis 数据结构

```
Key:   session:{session_id}
Value: JSON: {"history": [{role, content, timestamp}], "created_at": ISO8601}
TTL:   24 小时，每次 add_round 续期
```

## SQL Server 表结构

```sql
CREATE TABLE conversation_history (
    id BIGINT IDENTITY PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    role VARCHAR(16) NOT NULL,        -- 'user' | 'assistant'
    content NVARCHAR(MAX) NOT NULL,
    timestamp DATETIME2 DEFAULT GETDATE(),
    INDEX idx_session (session_id, timestamp)
);
```

## 写入策略

- `add_round()` 同步写 Redis + 异步写 SQL（BackgroundTasks）
- 异步写入失败不影响请求响应，打印 warning 日志
- SQL 写入异常时写入临时文件兜底（后续可加 Kafka/重试）

## 兼容性

- `session_id` 可选 → 不传则为无状态单次查询，行为不变
- 现有 API 请求/响应格式向后兼容（仅在 query 响应增加一个字段）
