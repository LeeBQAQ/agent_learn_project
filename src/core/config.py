import os
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

    # 多集合配置
    collections: dict[str, CollectionConfig] = field(
        default_factory=lambda: {
            "default": CollectionConfig(name="rag_documents", description="默认文档集合，存放通用知识", top_k=3),
            "programming": CollectionConfig(name="programming_docs", description="编程语言和技术文档", top_k=3),
            "ai_ml": CollectionConfig(name="ai_ml_docs", description="人工智能和机器学习相关文档", top_k=3),
            "frameworks": CollectionConfig(name="framework_docs", description="开发框架和工具文档", top_k=3),
        }
    )

    def get_milvus_connection_params(self) -> dict:
        params = {"uri": self.milvus_uri}
        if self.milvus_token:
            params["token"] = self.milvus_token
        if self.milvus_db_name:
            params["db_name"] = self.milvus_db_name
        return params

    def get_collection_config(self, collection_key: str = "default") -> CollectionConfig:
        if collection_key not in self.collections:
            raise ValueError(f"Collection '{collection_key}' 不存在，可用: {list(self.collections.keys())}")
        return self.collections[collection_key]

    @property
    def use_smart_routing(self) -> bool:
        return self.smart_mode in [SmartMode.ROUTING_ONLY, SmartMode.FULL]

    @property
    def use_smart_classification(self) -> bool:
        return self.smart_mode == SmartMode.FULL
