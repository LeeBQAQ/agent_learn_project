from typing import List, Optional, Tuple, Dict
from langchain_core.documents import Document
from module_02_config.config_01_rag_config import RAGConfig, CollectionConfig
from module_03_document.milvus_store import SimpleMilvusStore, get_milvus_client
from pymilvus import MilvusClient


class MultiCollectionRetriever:
    """支持多 Collection 的检索器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = get_milvus_client(config)
        
        # 缓存不同 collection 的 store 实例
        self._stores: Dict[str, SimpleMilvusStore] = {}

    def _get_store(self, collection_key: str, embeddings) -> SimpleMilvusStore:
        """获取或创建指定 collection 的 store 实例"""
        if collection_key not in self._stores:
            collection_config = self.config.get_collection_config(collection_key)
            
            self._stores[collection_key] = SimpleMilvusStore(
                milvus_client=self.client,
                collection_name=collection_config.name,
                embeddings=embeddings,
                config=self.config
            )
        
        return self._stores[collection_key]

    def retrieve(
        self, 
        query: str, 
        embeddings,
        collection_key: str = "default",
        k: Optional[int] = None
    ) -> List[Dict]:
        """从指定 collection 检索相关文档
        
        Args:
            query: 查询文本
            embeddings: 嵌入模型
            collection_key: collection 标识符
            k: 返回文档数量，默认为该 collection 配置的 top_k
            
        Returns:
            Document 列表
        """
        # 验证 collection_key 是否有效
        if collection_key not in self.config.collections:
            print(f"⚠️  Collection key '{collection_key}' 不存在，使用 default")
            collection_key = "default"
        
        store = self._get_store(collection_key, embeddings)
        coll_config = self.config.get_collection_config(collection_key)
        
        # 优先使用传入的 k，否则用该 collection 的配置
        k = k or coll_config.top_k
        
        return store.similarity_search(query=query, k=k)

    def retrieve_from_multiple(
        self,
        query: str,
        embeddings,
        collection_keys: List[str],
        k_per_collection: Optional[int] = None
    ) -> Dict[str, List[Document]]:
        """从多个 collection 并行检索
        
        Args:
            query: 查询文本
            embeddings: 嵌入模型
            collection_keys: collection 标识符列表
            k_per_collection: 每个 collection 返回的文档数
            
        Returns:
            {collection_key: [documents]} 字典
        """
        results = {}
        for key in collection_keys:
            try:
                docs = self.retrieve(query, embeddings, key, k_per_collection)
                results[key] = docs
            except Exception as e:
                print(f"⚠️ Collection '{key}' 检索失败: {e}")
                results[key] = []
        
        return results

    def retrieve_merged(
        self,
        query: str,
        embeddings,
        collection_keys: List[str],
        total_k: int = 5
    ) -> List[Document]:
        """从多个 collection 检索并合并结果（按分数排序）
        
        Args:
            query: 查询文本
            embeddings: 嵌入模型
            collection_keys: collection 标识符列表
            total_k: 总共返回的文档数
            
        Returns:
            合并后的 Document 列表（按相似度排序）
        """
        all_docs_with_scores = []
        
        # 从每个 collection 检索
        for key in collection_keys:
            try:
                store = self._get_store(key, embeddings)
                docs = store.similarity_search(query=query, k=total_k)
                
                for doc in docs:
                    score = doc.metadata.get("score", 0.0)
                    # 添加来源标记
                    doc.metadata["collection"] = key
                    all_docs_with_scores.append((doc, score))
            except Exception as e:
                print(f"⚠️ Collection '{key}' 检索失败: {e}")
        
        # 按分数排序，取前 total_k 个
        all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in all_docs_with_scores[:total_k]]


# 保持向后兼容的简单 Retriever
class Retriever:
    """检索器：从向量存储中检索相关文档"""

    def __init__(self, vector_store: SimpleMilvusStore, config: RAGConfig):
        self.vector_store = vector_store
        self.config = config

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """检索相关文档
        
        Args:
            query: 查询文本
            k: 返回文档数量，默认为配置中的 top_k
            
        Returns:
            Document 列表
        """
        k = k or self.config.top_k
        return self.vector_store.similarity_search(query=query, k=k)

    def retrieve_with_scores(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """检索文档并返回相似度分数
        
        Args:
            query: 查询文本
            k: 返回文档数量，默认为配置中的 top_k
            
        Returns:
            (Document, score) 元组列表，score 为相似度分数
        """
        k = k or self.config.top_k
        docs = self.vector_store.similarity_search(query=query, k=k)
        
        # 从 metadata 中提取分数
        result = []
        for doc in docs:
            score = doc.metadata.get("score", 0.0)
            result.append((doc, score))
        
        return result
    
    def as_langchain_retriever(self, k: Optional[int] = None):
        """返回兼容 LangChain 的 retriever 对象
        
        Args:
            k: 返回文档数量
            
        Returns:
            可调用的 retriever 函数
        """
        return self.vector_store.as_retriever(k=k or self.config.top_k)