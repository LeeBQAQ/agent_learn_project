from langchain_core.output_parsers import StrOutputParser

from module_02_config.config_01_rag_config import RAGConfig
from model.model_init import model
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict
from .template_library import TemplateLibrary


class Generator:
    """生成器：基于上下文生成回答"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.model = model
        # rag提示词
        self.rag_prompt = TemplateLibrary.RAG_PROMPT
        # 查询改写提示
        self.rewrite_prompt = TemplateLibrary.REWRITE_PROMPT
        self.eval_prompt = TemplateLibrary.EVAL_PROMPT

    def rewrite_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """根据对话历史改写查询"""
        if not chat_history:
            return query

        # 转换对话历史格式
        messages = []
        for msg in chat_history[-4:]:  # 只用最近4论对话
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        # 管道：提示词 | 模型 | 输出解析器
        chain = self.rewrite_prompt | self.model | StrOutputParser()
        return chain.invoke({
            "query": query,
            "chat_history": messages
        })

    def generate(self, query: str, context: str, chat_history: List[Dict[str, str]] = None) -> float:
        """使用改写后的查询，生成回答"""
        messages = []
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        chain = self.rag_prompt | self.model | StrOutputParser()
        return chain.invoke({
            "query": query,
            "context": context,
            "chat_history": messages
        })

    def evaluate_confidence(self, query: str, context: str, answer: str) -> float:
        """评估回答的置信度"""
        chain = self.eval_prompt | self.model | StrOutputParser()
        try:
            score = float(chain.invoke({
                "query": query,
                "context": context,
                "answer": answer}).strip())
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5
