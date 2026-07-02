import logging
from typing import List, Optional
from pymilvus import MilvusClient
from langchain_core.documents import Document
from src.core.config import RAGConfig

logger = logging.getLogger(__name__)


class SimpleMilvusStore:
    """Milvus 向量存储封装"""

    def __init__(self, milvus_client: MilvusClient, collection_name: str, embeddings, config: RAGConfig = None):
        self.client = milvus_client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.config = config

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """相似度搜索"""
        k = k or (self.config.top_k if self.config else 3)
        query_embedding = self.embeddings.embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=k,
            output_fields=["text", "source"],
        )
        docs = []
        for hit_list in results:
            for hit in hit_list:
                entity = hit.get("entity", {})
                docs.append(Document(
                    page_content=entity.get("text", ""),
                    metadata={"source": entity.get("source", ""), "score": hit.get("distance", 0)},
                ))
        return docs

    def as_retriever(self, k: Optional[int] = None):
        """返回兼容 LangChain 的 retriever 函数"""
        k = k or (self.config.top_k if self.config else 3)

        def retriever(query: str) -> str:
            docs = self.similarity_search(query, k=k)
            return "\n\n".join([doc.page_content for doc in docs])

        return retriever

    def bm25_search(self, query: str, k: int = 3) -> List[Document]:
        """BM25 关键词检索（Milvus TEXT_MATCH）"""
        try:
            safe_query = query.replace('"', '\\"')
            results = self.client.query(
                collection_name=self.collection_name,
                filter=f'TEXT_MATCH("text", "{safe_query}")',
                output_fields=["text", "source"],
                limit=k,
            )
            docs = []
            for entity in results:
                docs.append(Document(
                    page_content=entity.get("text", ""),
                    metadata={"source": entity.get("source", ""), "score": 0.0},
                ))
            return docs
        except Exception as e:
            logger.warning("BM25 检索失败: %s", e)
            return []

    def hybrid_search(self, query: str, k: int = 3) -> List[Document]:
        """混合检索：向量 + BM25 + RRF 融合"""
        fetch_k = k * 2

        vector_docs = self.similarity_search(query, k=fetch_k)
        bm25_docs = self.bm25_search(query, k=fetch_k)

        return _rrf_fusion(vector_docs, bm25_docs, k=60, top_k=k)


def _rrf_fusion(
    vector_docs: List[Document],
    bm25_docs: List[Document],
    k: int = 60,
    top_k: int = 3,
) -> List[Document]:
    """RRF (Reciprocal Rank Fusion) 融合两路检索结果"""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(vector_docs, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in doc_map:
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_k]]


def get_milvus_client(config: RAGConfig) -> MilvusClient:
    """创建 Milvus 客户端"""
    return MilvusClient(**config.get_milvus_connection_params())
