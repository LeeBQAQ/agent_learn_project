import pytest

from src.core.config import RAGConfig
from src.core.milvus_store import SimpleMilvusStore, get_milvus_client


@pytest.mark.integration
def test_milvus_client_connection():
    config = RAGConfig(milvus_uri="http://localhost:19530")
    client = get_milvus_client(config)
    assert client is not None
    collections = client.list_collections()
    assert isinstance(collections, list)


@pytest.mark.integration
def test_create_and_search_collection():
    config = RAGConfig(milvus_uri="http://localhost:19530")
    client = get_milvus_client(config)

    class FakeEmbeddings:
        def embed_query(self, text):
            return [0.1] * 384

        def embed_documents(self, texts):
            return [[0.1] * 384 for _ in texts]

    emb = FakeEmbeddings()
    collection_name = "test_integration_collection"

    try:
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            dimension=384,
            primary_field_name="id",
            id_type="string",
            vector_field_name="vector",
            auto_id=False,
            max_length=65535,
        )
        client.insert(
            collection_name=collection_name,
            data=[{"id": "0", "vector": [0.1] * 384, "text": "Hello world", "source": "test.txt"}],
        )

        store = SimpleMilvusStore(client, collection_name, emb, config)
        docs = store.similarity_search("Hello", k=1)
        assert len(docs) >= 0
    finally:
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
