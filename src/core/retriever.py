
from langchain_core.documents import Document

from src.core.config import RAGConfig
from src.core.milvus_store import SimpleMilvusStore, get_milvus_client


class MultiCollectionRetriever:
    """支持多 Collection 的检索器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = get_milvus_client(config)
        self._stores: dict[str, SimpleMilvusStore] = {}

    def _get_store(self, collection_key: str, embeddings) -> SimpleMilvusStore:
        if collection_key not in self._stores:
            coll_config = self.config.get_collection_config(collection_key)
            self._stores[collection_key] = SimpleMilvusStore(
                milvus_client=self.client, collection_name=coll_config.name, embeddings=embeddings, config=self.config
            )
        return self._stores[collection_key]

    def retrieve(
        self, query: str, embeddings, collection_key: str = "default", k: int | None = None
    ) -> list[Document]:
        if collection_key not in self.config.collections:
            collection_key = "default"
        store = self._get_store(collection_key, embeddings)
        coll_config = self.config.get_collection_config(collection_key)
        k = k or coll_config.top_k
        if self.config.hybrid_search:
            return store.hybrid_search(query=query, k=k)
        return store.similarity_search(query=query, k=k)

    def retrieve_from_multiple(
        self, query: str, embeddings, collection_keys: list[str], k_per_collection: int | None = None
    ) -> dict[str, list[Document]]:
        results = {}
        for key in collection_keys:
            try:
                results[key] = self.retrieve(query, embeddings, key, k_per_collection)
            except Exception as e:
                print(f"Warning: Collection '{key}' 检索失败: {e}")
                results[key] = []
        return results

    def retrieve_merged(self, query: str, embeddings, collection_keys: list[str], total_k: int = 5) -> list[Document]:
        all_docs_with_scores = []
        for key in collection_keys:
            try:
                store = self._get_store(key, embeddings)
                docs = store.similarity_search(query=query, k=total_k)
                for doc in docs:
                    doc.metadata["collection"] = key
                    all_docs_with_scores.append((doc, doc.metadata.get("score", 0.0)))
            except Exception as e:
                print(f"Warning: Collection '{key}' 检索失败: {e}")
        all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in all_docs_with_scores[:total_k]]


class Retriever:
    """单 Collection 检索器"""

    def __init__(self, vector_store: SimpleMilvusStore, config: RAGConfig):
        self.vector_store = vector_store
        self.config = config

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        k = k or self.config.top_k
        return self.vector_store.similarity_search(query=query, k=k)

    def retrieve_with_scores(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        k = k or self.config.top_k
        docs = self.vector_store.similarity_search(query=query, k=k)
        return [(doc, doc.metadata.get("score", 0.0)) for doc in docs]

    def as_langchain_retriever(self, k: int | None = None):
        return self.vector_store.as_retriever(k=k or self.config.top_k)
