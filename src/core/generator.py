from collections.abc import Iterator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.utils import Output

from src.core.config import RAGConfig
from src.core.model import model
from src.core.templates import TemplateLibrary


class Generator:
    """生成器：基于上下文生成回答"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.model = model
        self.rag_prompt = TemplateLibrary.RAG_PROMPT
        self.rewrite_prompt = TemplateLibrary.REWRITE_PROMPT
        self.eval_prompt = TemplateLibrary.EVAL_PROMPT

    def rewrite_query(self, query: str, chat_history: list[dict[str, str]]) -> str:
        """根据对话历史改写查询"""
        if not chat_history:
            return query
        messages = []
        for msg in chat_history[-4:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        chain = self.rewrite_prompt | self.model | StrOutputParser()
        return chain.invoke({"query": query, "chat_history": messages})

    def generate(self, query: str, context: str, chat_history: list[dict[str, str]] = None) -> Output:
        """生成回答"""
        messages = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
        chain = self.rag_prompt | self.model | StrOutputParser()
        return chain.invoke({"query": query, "context": context, "chat_history": messages})

    def generate_chat(self, query: str, chat_history: list[dict[str, str]] = None) -> str:
        """无知识库时直接用大模型回答"""
        messages = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
        chain = TemplateLibrary.CHAT_PROMPT | self.model | StrOutputParser()
        return chain.invoke({"query": query, "chat_history": messages})

    def stream_generate(self, query: str, context: str, chat_history: list[dict[str, str]] = None) -> Iterator[str]:
        """流式生成，逐 token yield"""
        messages = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

        prompt = self.rag_prompt if context else TemplateLibrary.CHAT_PROMPT
        chain = prompt | self.model

        for chunk in chain.stream({"query": query, "context": context, "chat_history": messages}):
            if chunk.content:
                yield chunk.content

    def evaluate_confidence(self, query: str, context: str, answer: str) -> float:
        """评估回答置信度"""
        chain = self.eval_prompt | self.model | StrOutputParser()
        try:
            score = float(chain.invoke({"query": query, "context": context, "answer": answer}).strip())
            return min(max(score, 0.0), 1.0)
        except (ValueError, TypeError):
            return 0.5
