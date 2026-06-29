from module_02_config.config_01_rag_config import RAGConfig
from model_init import model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict

class Generator:
    """生成器：基于上下文生成回答"""

    def __init__(self, config: RAGConfig):
        self.config = RAGConfig()
        self.model = model

        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的问答助手。请基于提供的上下文信息回答用户的问题。

        重要规则：
        1. 只使用提供的上下文信息来回答问题
        2. 如果上下文中没有相关信息，请诚实地说"根据提供的信息，我无法回答这个问题"
        3. 回答要准确、简洁、有条理
        4. 在回答末尾标注信息来源

        上下文信息：
        {context}
        """),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{query}")
        ])

        # 查询改写提示
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个查询优化专家。请根据对话历史，将用户的问题改写为一个独立、完整的查询。

        如果问题本身已经很清晰完整，直接返回原问题。
        只返回改写后的查询，不要添加任何解释。"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "原始问题：{query}\n\n请改写为独立完整的查询：")
        ])

    def rewrite_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """根据对话历史改写查询"""
        if not chat_history:
            return query

        # 转换对话历史格式