import pytest
from src.core.config import RAGConfig


@pytest.fixture
def test_config() -> RAGConfig:
    return RAGConfig(chunk_size=100, chunk_overlap=20, top_k=2)


@pytest.fixture
def sample_documents():
    return [
        "Python 是一种广泛使用的编程语言，支持面向对象和函数式编程。",
        "神经网络是深度学习的基础，由多层神经元组成。",
        "Django 是一个 Python Web 框架，遵循 MTV 模式。",
    ]
