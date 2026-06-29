from pymilvus import MilvusClient
from typing import List, Optional, Dict
from langchain_core.documents import Document
from module_02_config.config_01_rag_config import RAGConfig


class SimpleMilvusStore:

    def __init__(self, milvus_client: MilvusClient, collection_name: str, embeddings, config: RAGConfig = None):
        self.client = milvus_client  # Milvus 客户端连接
        self.collection_name = collection_name  # 集合名称
        self.embeddings = embeddings  # 向量模型
        self.config = config

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """相似度搜索，返回 Document 列表"""
        k = k or (self.config.top_k if self.config else 3)
        query_embedding = self.embeddings.embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_embeddings=[query_embedding],
            limit=k,
            output_fields=["text", "source"],
        )
        docs = []
        for hit_list in results:
            for hit in hit_list:
                entity = hit.get("entity", {})
                docs.append(
                    Document(
                        page_content=entity.get("text", ""),
                        metadata={
                            # "source": entity.get("source", ""),
                            "score": hit.get("distance", 0)},
                    )
                )
        return docs

    def as_retriever(self, k: Optional[int] = None):
        """返回一个 retriever 函数，兼容 Agent 工具使用"""
        k = k or (self.config.top_k if self.config else 3)

        def retriever(query: str) -> str:
            docs = self.similarity_search(query, k=k)
            return "\n\n".join([doc.page_content for doc in docs])

        return retriever


def get_milvus_client(config: RAGConfig):
    """根据配置创建 Milvus 客户端"""
    return MilvusClient(**config.get_milvus_connection_params())
