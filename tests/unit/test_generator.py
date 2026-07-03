from unittest.mock import MagicMock, patch

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


def test_stream_generate_yields_tokens():
    gen = Generator()
    mock_chain = MagicMock()
    mock_chunk1 = MagicMock()
    mock_chunk1.content = "RAG"
    mock_chunk4 = MagicMock()
    mock_chunk4.content = "。"

    mock_chain.stream.return_value = [mock_chunk1, mock_chunk4]

    with patch.object(gen, "rag_prompt") as mock_prompt:
        mock_prompt.__or__.return_value = mock_chain
        tokens = list(gen.stream_generate("什么是RAG？", context="RAG是检索增强生成", chat_history=None))

    assert tokens == ["RAG", "。"]


def test_stream_generate_empty_chunk_skipped():
    gen = Generator()
    mock_chain = MagicMock()
    mock_chunk1 = MagicMock()
    mock_chunk1.content = ""
    mock_chunk2 = MagicMock()
    mock_chunk2.content = "有效内容"

    mock_chain.stream.return_value = [mock_chunk1, mock_chunk2]

    with patch.object(gen, "rag_prompt") as mock_prompt:
        mock_prompt.__or__.return_value = mock_chain
        tokens = list(gen.stream_generate("测试", context="ctx"))

    assert tokens == ["有效内容"]

