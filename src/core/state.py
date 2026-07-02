from typing import Any, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    """RAG 流程状态"""

    query: str
    chat_history: list[dict[str, str]]
    documents: list[Document]
    context: str
    answer: str
    sources: list[dict[str, Any]]
    confidence: float
