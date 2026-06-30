from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class TemplateLibrary:
    """提示词模板库"""

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的问答助手。请基于提供的上下文信息回答用户的问题。
重要规则：
  1. 只使用提供的上下文信息来回答问题
  2. 如果上下文中没有相关信息，请诚实地说"根据提供的信息，我无法回答这个问题"
  3. 回答要准确、简洁、有条理
  4. 在回答末尾标注信息来源
上下文信息：
{context}"""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{query}")
    ])

    REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是一个查询优化专家。请根据对话历史，将用户的问题改写为一个独立、完整的查询。
如果问题本身已经很清晰完整，直接返回原问题。
只返回改写后的查询，不要添加任何解释。"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "原始问题：{query}\n\n请改写为独立完整的查询：")
    ])

    EVAL_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """评估以下回答的置信度。考虑：
  1. 回答是否基于提供的上下文
  2. 信息的相关性和准确性
  3. 回答的完整性
只返回一个0到1之间的数字，表示置信度。"""),
        ("human", """上下文：{context}
问题：{query}
回答：{answer}
置信度:（0-1）""")
    ])

    COLLECTION_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能路由专家。根据用户的问题，判断应该从哪个数据集合中检索信息。

可用的数据集合：
{collections_info}

请分析问题的主题和意图，选择最相关的一个或多个集合。

返回格式：JSON 数组，包含选中的集合名称
例如：["collection1", "collection2"]

如果不确定，可以选择多个相关集合或返回所有集合。"""),
        ("human", "用户问题：{query}\n\n请选择合适的数据集合：")
    ])

    DOCUMENT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是一个文档分类专家。分析文档内容，判断它应该存储到哪个数据集合。

可用的数据集合：
{collections_info}

请仔细阅读文档内容，选择最匹配的集合。

返回格式：JSON 对象
{{
    "collection": "集合名称",
    "confidence": 0.95,
    "reason": "简短的分类理由"
}}"""),
        ("human", "文档内容：\n{document_content}\n\n请分类到合适的集合：")
    ])
