from typing import List, Dict, Any, TypedDict
from langchain_core.documents import Document


class RAGState(TypedDict):
    """RAG 流程状态"""
    query: str
    chat_history: List[Dict[str, str]]
    documents: List[Document]
    context: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
