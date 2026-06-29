from dataclasses import dataclass, field
from typing import Optional, Dict


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

    """Milvus 配置"""
    # Milvus 向量数据库配置
    milvus_uri: str = "http://localhost:19530"  # Milvus 连接 URI（推荐方式）
    milvus_token: Optional[str] = None  # Milvus 认证 token（可选）
    milvus_db_name: str = "default"  # 数据库名称

    # 集合配置
    collection_name: str = "rag_documents"  # 默认集合名称
    metric_type: str = "COSINE"  # 距离度量类型：COSINE, L2, IP
    auto_id: bool = False  # 是否自动生成 ID

    # 索引配置
    index_type: str = "AUTOINDEX"  # 索引类型
    nlist: int = 128  # IVF 索引的聚类中心数

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