import json
import logging
from datetime import datetime, timezone
from typing import Any

import redis

from src.core.config import RAGConfig

logger = logging.getLogger(__name__)


class SessionStore:
    """会话记忆存储：Redis 热数据"""

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
            return history[-self.config.session_max_rounds * 2:]
        except Exception as e:
            logger.warning("获取会话历史失败: %s", e)
            return []

    def add_round(self, session_id: str, role: str, content: str) -> None:
        """添加一轮对话：同步写 Redis"""
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
                max_msgs = self.config.session_max_rounds * 2
                session["history"] = session["history"][-max_msgs:]

                ttl_seconds = self.config.session_ttl_hours * 3600
                self._redis.setex(key, ttl_seconds, json.dumps(session, ensure_ascii=False))
            except Exception as e:
                logger.warning("Redis 写入失败: %s", e)

    def get_full_history(self, session_id: str) -> list[dict[str, Any]]:
        """获取全量历史 (回退到 Redis)"""
        return self.get_history(session_id)

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
