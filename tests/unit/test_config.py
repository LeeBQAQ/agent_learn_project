from src.core.config import RAGConfig, _collection_name_to_key, sanitize_collection_name


def test_hybrid_search_default_false():
    config = RAGConfig()
    assert config.hybrid_search is False


def test_hybrid_search_can_enable():
    config = RAGConfig(hybrid_search=True)
    assert config.hybrid_search is True


def test_collections_empty_by_default():
    config = RAGConfig()
    assert config.collections == {}


def test_ensure_default_collection():
    config = RAGConfig()
    config.ensure_default_collection()
    assert "default" in config.collections
    assert config.collections["default"].name == "rag_documents"


def test_register_collection():
    config = RAGConfig()
    config.register_collection("test", "rag_test", "测试")
    assert "test" in config.collections
    assert config.collections["test"].name == "rag_test"


def test_has_collection():
    config = RAGConfig()
    config.register_collection("db", "rag_db")
    assert config.has_collection("db") is True
    assert config.has_collection("nonexistent") is False


def test_sanitize_collection_name_chinese():
    assert sanitize_collection_name("编程") == "default"


def test_sanitize_collection_name_valid():
    assert sanitize_collection_name("programming") == "programming"


def test_sanitize_collection_name_special_chars():
    assert sanitize_collection_name("ai/ml") == "ai_ml"


def test_collection_name_to_key():
    assert _collection_name_to_key("rag_documents") == "documents"
    assert _collection_name_to_key("programming_docs") == "programming"
