from src.core.config import RAGConfig


def test_hybrid_search_default_false():
    config = RAGConfig()
    assert config.hybrid_search is False


def test_hybrid_search_can_enable():
    config = RAGConfig(hybrid_search=True)
    assert config.hybrid_search is True
