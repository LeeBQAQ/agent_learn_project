from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum


class SmartMode(Enum):
    """智能功能模式"""
    DISABLED = "disabled"          # 禁用所有智能功能
    ROUTING_ONLY = "routing_only"  # 仅启用查询路由
    FULL = "full"                  # 启用所有智能功能（路由+分类）


@dataclass
class CollectionConfig:
    """单个 Collection 的配置"""
    name: str  # collection 名称
    description: str = ""  # 描述用途
    top_k: int = 3  # 该 collection 的默认检索数量
    metric_type: str = "COSINE"
    auto_id: bool = False


@dataclass
class RAGConfig:
    """RAG 系统配置"""

    # 模型配置
    temperature: float = 0.1

    # 分块配置
    # 块大小，太大包含无用信息，降低精度。太小，丢失上下文，llm无法理解。
    chunk_size: int = 512
    # 块重叠，保持语义连贯，避免中重要信息被切断
    chunk_overlap: int = 100

    # 检索配置
    top_k: int = 3  # 每次查询时，返回最相关的 top_K 个文档
    search_type: str = "similarity"  # 检索算法
    # similarity - 相似度搜索：基于余弦相似度或欧氏距离，找到与查询向量最接近的文档）
    # mmr - 最大边界相关度搜索：基于文档的概率分布，找到与查询向量最相关的文档

    # 生成配置
    max_tokens: int = 1024 # 最大输出token，防止生成超长回答，费用失控

    # 智能功能配置
    smart_mode: SmartMode = SmartMode.FULL  # 智能功能模式

    """Milvus 配置"""
    # Milvus 向量数据库配置
    milvus_uri: str = "http://localhost:19530"  # Milvus 连接 URI（推荐方式）
    milvus_token: Optional[str] = None  # Milvus 认证 token（可选）
    milvus_db_name: str = "default"  # 数据库名称

    # 多集合配置
    collections: Dict[str, CollectionConfig] = field(default_factory=lambda: {
        "default": CollectionConfig(
            name="rag_documents", 
            description="默认文档集合，存放通用知识",
            top_k=3
        ),
        "programming": CollectionConfig(
            name="programming_docs",
            description="编程语言和技术文档，包括 Python、Java 等编程语言相关知识",
            top_k=3
        ),
        "ai_ml": CollectionConfig(
            name="ai_ml_docs",
            description="人工智能和机器学习相关文档，包括算法、框架、概念等",
            top_k=3
        ),
        "frameworks": CollectionConfig(
            name="framework_docs",
            description="开发框架和工具文档，包括 LangChain、Django、Flask 等框架",
            top_k=3
        ),
    })

    def get_milvus_connection_params(self) -> dict:
        """获取 Milvus 连接参数"""
        params = {
            "uri": self.milvus_uri,
        }
        if self.milvus_token:
            params["token"] = self.milvus_token
        if self.milvus_db_name:
            params["db_name"] = self.milvus_db_name
        return params
    
    def get_collection_config(self, collection_key: str = "default") -> CollectionConfig:
        """获取指定 collection 的配置"""
        if collection_key not in self.collections:
            raise ValueError(f"Collection '{collection_key}' 不存在，可用: {list(self.collections.keys())}")
        return self.collections[collection_key]
    
    @property
    def use_smart_routing(self) -> bool:
        """是否启用智能查询路由"""
        return self.smart_mode in [SmartMode.ROUTING_ONLY, SmartMode.FULL]
    
    @property
    def use_smart_classification(self) -> bool:
        """是否启用智能文档分类"""
        return self.smart_mode == SmartMode.FULL