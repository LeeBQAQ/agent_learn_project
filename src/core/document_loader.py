from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import RAGConfig
from src.core.embeddings import get_embeddings
from src.core.milvus_store import SimpleMilvusStore, get_milvus_client
from src.core.smart_router import DocumentClassifier


class DocumentProcessor:
    """文档处理器：加载、分块、分类、向量化存储"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )
        self.embeddings = get_embeddings()
        self.use_smart_classification = config.use_smart_classification
        self.classifier = DocumentClassifier(config) if self.use_smart_classification else None

    def load_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> list[Document]:
        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {"source": f"doc_{i}"}
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return list(self.text_splitter.split_documents(documents))

    def classify_and_store(
        self, documents: list[Document], default_collection: str = "default"
    ) -> dict[str, list[Document]]:
        classified: dict[str, list[Document]] = {}
        for doc in documents:
            if self.use_smart_classification and self.classifier:
                result = self.classifier.classify_document(doc.page_content)
                collection = result.get("collection", default_collection)
                if collection not in self.config.collections:
                    collection = default_collection
            else:
                collection = default_collection
            classified.setdefault(collection, []).append(doc)
        return classified

    def create_vector_store(self, documents: list[Document], milvus_collection: str) -> SimpleMilvusStore:
        texts = [doc.page_content for doc in documents]
        sources = [doc.metadata.get("source", "") for doc in documents]
        vectors = self.embeddings.embed_documents(texts)
        dim = len(vectors[0])

        client = get_milvus_client(self.config)
        if not client.has_collection(milvus_collection):
            create_params = dict(
                collection_name=milvus_collection,
                dimension=dim,
                primary_field_name="id",
                id_type="string",
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False,
                max_length=65535,
            )
            if self.config.hybrid_search:
                create_params["enable_dynamic_field"] = True
            client.create_collection(**create_params)
        data = [{"id": str(i), "vector": vec, "text": texts[i], "source": sources[i]} for i, vec in enumerate(vectors)]
        client.insert(collection_name=milvus_collection, data=data)
        return SimpleMilvusStore(client, milvus_collection, self.embeddings, self.config)

    def process(
        self, texts: list[str], metadatas: list[dict] | None = None, milvus_collection: str = "documents"
    ) -> dict[str, SimpleMilvusStore]:
        documents = self.load_documents(texts, metadatas)
        classified = self.classify_and_store(documents, milvus_collection)
        vector_stores: dict[str, SimpleMilvusStore] = {}
        for collection_key, docs in classified.items():
            coll_config = self.config.get_collection_config(collection_key)
            chunks = self.split_documents(docs)
            vector_stores[collection_key] = self.create_vector_store(chunks, coll_config.name)
        return vector_stores
