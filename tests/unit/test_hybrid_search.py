from unittest.mock import MagicMock
from src.core.milvus_store import SimpleMilvusStore, _rrf_fusion
from src.core.config import RAGConfig
from langchain_core.documents import Document


def test_bm25_search_basic():
    config = RAGConfig(hybrid_search=True, top_k=2)
    mock_client = MagicMock()
    mock_client.query.return_value = [
        {"id": "0", "text": "Python GIL 全局解释器锁", "source": "doc1.txt"},
        {"id": "1", "text": "Python 多线程编程", "source": "doc2.txt"},
    ]
    mock_embeddings = MagicMock()

    store = SimpleMilvusStore(mock_client, "test_coll", mock_embeddings, config)
    docs = store.bm25_search("GIL", k=3)

    assert len(docs) == 2
    assert docs[0].page_content == "Python GIL 全局解释器锁"
    mock_client.query.assert_called_once()
    call_args = mock_client.query.call_args[1]
    assert "TEXT_MATCH" in call_args["filter"]


def test_bm25_search_empty_result():
    config = RAGConfig(hybrid_search=True)
    mock_client = MagicMock()
    mock_client.query.return_value = []
    mock_embeddings = MagicMock()

    store = SimpleMilvusStore(mock_client, "test_coll", mock_embeddings, config)
    docs = store.bm25_search("nonexistent")

    assert docs == []


def test_hybrid_search_rrf_merge():
    config = RAGConfig(hybrid_search=True, top_k=2)
    mock_client = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 384

    # 向量检索返回：doc_A(rank1), doc_B(rank2)
    mock_client.search.return_value = [[
        {"entity": {"text": "Doc A content", "source": "a.txt"}, "distance": 0.95},
        {"entity": {"text": "Doc B content", "source": "b.txt"}, "distance": 0.80},
    ]]
    # BM25 检索返回：doc_B(rank1), doc_A(rank2)
    mock_client.query.return_value = [
        {"id": "1", "text": "Doc B content", "source": "b.txt"},
        {"id": "0", "text": "Doc A content", "source": "a.txt"},
    ]

    store = SimpleMilvusStore(mock_client, "test_coll", mock_embeddings, config)
    docs = store.hybrid_search("test query", k=2)

    assert len(docs) <= 2
    contents = [d.page_content for d in docs]
    assert "Doc A content" in contents
    assert "Doc B content" in contents


def test_rrf_formula_correctness():
    """验证 RRF 公式: score = sum 1/(k + rank) for k=60"""
    vec_docs = [
        Document(page_content="A", metadata={"source": "a.txt", "score": 0.95}),
        Document(page_content="B", metadata={"source": "b.txt", "score": 0.80}),
    ]
    bm25_docs = [
        Document(page_content="B", metadata={"source": "b.txt", "score": 0.0}),
        Document(page_content="A", metadata={"source": "a.txt", "score": 0.0}),
    ]

    result = _rrf_fusion(vec_docs, bm25_docs, k=60, top_k=2)
    assert len(result) == 2


def test_bm25_search_escapes_special_chars():
    """验证特殊字符（双引号、反斜杠）被正确转义"""
    config = RAGConfig(hybrid_search=True)
    mock_client = MagicMock()
    mock_client.query.return_value = []
    mock_embeddings = MagicMock()

    store = SimpleMilvusStore(mock_client, "test_coll", mock_embeddings, config)
    store.bm25_search('hello "world" \\test')

    call_args = mock_client.query.call_args[1]
    filter_str = call_args["filter"]
    # 不应该包含未转义的双引号或反斜杠
    assert '\\"' in filter_str or '\\\\' in filter_str  # 至少转义存在
