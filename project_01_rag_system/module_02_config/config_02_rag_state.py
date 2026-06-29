
from typing import List, Dict, Any
from langchain_core.documents import Document


class RAGState:
    """RAG 流程状态"""
    chat_history: List[Dict[str, str]]  # 对话历史
    query: str  # 用户查询的内容

    documents: List[Document]  # 检索到的内容
    context:str  # 格式化的上下文
    answer: str  # 回答
    sources: List[Dict[str, Any]] # 来源信息

    confidence: float # 置信度
