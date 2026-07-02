from unittest.mock import patch

from src.core.generator import Generator


def test_rewrite_query_no_history():
    gen = Generator()
    result = gen.rewrite_query("什么是 RAG？", [])
    assert result == "什么是 RAG？"


@patch("src.core.generator.model")
def test_evaluate_confidence_bounds(mock_model):
    gen = Generator()
    # evaluate_confidence 在无法解析 LLM 返回值时返回 0.5
    result = gen.evaluate_confidence("q", "ctx", "ans")
    assert 0.0 <= result <= 1.0
