import os
import re
from dataclasses import dataclass, field
from enum import Enum


class SmartMode(Enum):
    """智能功能模式"""

    DISABLED = "disabled"
    ROUTING_ONLY = "routing_only"
    FULL = "full"


@dataclass
class CollectionConfig:
    """单个 Collection 的配置"""

    name: str
    description: str = ""
    top_k: int = 3
    metric_type: str = "COSINE"
    auto_id: bool = False


def sanitize_collection_name(name: str) -> str:
    """确保名称合法：仅字母数字下划线"""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = name[:255].strip("_")
    return name or "default"


def _collection_name_to_key(name: str) -> str:
    """Milvus collection name → 简短 key，用于路由"""
    return name.replace("_docs", "").replace("rag_", "")


@dataclass
class RAGConfig:
    """RAG 系统配置"""

    # 模型配置
    temperature: float = 0.1

    # 分块配置
    chunk_size: int = 512
    chunk_overlap: int = 100

    # 检索配置
    top_k: int = 3
    search_type: str = "similarity"

    # 生成配置
    max_tokens: int = 1024

    # 混合检索配置
    hybrid_search: bool = False

    # 启动初始化
    eager_init: bool = True

    # 会话记忆配置
    redis_uri: str = field(default_factory=lambda: os.getenv("REDIS_URI", "redis://localhost:6379/0"))
    sqlserver_conn: str = field(default_factory=lambda: os.getenv("SQLSERVER_CONN", ""))
    session_ttl_hours: int = 24
    session_max_rounds: int = 10

    # 智能路由模式
    smart_mode: SmartMode = SmartMode.FULL

    # Milvus 配置
    milvus_uri: str = field(default_factory=lambda: os.getenv("MILVUS_URI", "http://localhost:19530"))
    milvus_token: str | None = field(default_factory=lambda: os.getenv("MILVUS_TOKEN"))
    milvus_db_name: str = field(default_factory=lambda: os.getenv("MILVUS_DB_NAME", "default"))

    # 多集合配置（启动时从 Milvus 自动同步）
    collections: dict[str, CollectionConfig] = field(default_factory=dict)

    def get_milvus_connection_params(self) -> dict:
        params = {"uri": self.milvus_uri}
        if self.milvus_token:
            params["token"] = self.milvus_token
        if self.milvus_db_name:
            params["db_name"] = self.milvus_db_name
        return params

    def ensure_default_collection(self) -> str:
        """确保 default 集合存在，返回其 key"""
        if "default" not in self.collections:
            self.collections["default"] = CollectionConfig(
                name="rag_documents", description="默认文档集合", top_k=3
            )
        return "default"

    def sync_collections(self, client) -> list[str]:
        """从 Milvus 同步实际存在的集合到内存，返回新增的 key 列表"""
        added = []
        self.ensure_default_collection()
        for coll_name in client.list_collections():
            key = _collection_name_to_key(coll_name)
            if key not in self.collections:
                self.collections[key] = CollectionConfig(name=coll_name, description="")
                added.append(key)
        return added

    def register_collection(self, key: str, name: str, description: str = "") -> None:
        """注册新集合到内存"""
        if key not in self.collections:
            self.collections[key] = CollectionConfig(name=name, description=description)

    def has_collection(self, key: str) -> bool:
        return key in self.collections

    def get_collection_config(self, collection_key: str = "default") -> CollectionConfig:
        if collection_key not in self.collections:
            self.ensure_default_collection()
            if collection_key not in self.collections:
                raise ValueError(f"Collection '{collection_key}' 不存在，可用: {list(self.collections.keys())}")
        return self.collections[collection_key]

    @property
    def use_smart_routing(self) -> bool:
        return self.smart_mode in [SmartMode.ROUTING_ONLY, SmartMode.FULL]

    @property
    def use_smart_classification(self) -> bool:
        return self.smart_mode == SmartMode.FULL
