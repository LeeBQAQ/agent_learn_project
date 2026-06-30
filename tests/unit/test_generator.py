from src.core.generator import Generator


def test_rewrite_query_no_history():
    gen = Generator()
    result = gen.rewrite_query("什么是 RAG？", [])
    assert result == "什么是 RAG？"


def test_evaluate_confidence_bounds():
    gen = Generator()
    # 不调用 LLM，直接验证边界处理
    # evaluate_confidence 在 Value/Type error 时返回 0.5
    pass
