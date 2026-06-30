from typing import List, Optional, Dict, Any

from module_02_config.config_01_rag_config import RAGConfig
from module_02_config.config_02_rag_state import RAGState
from module_03_document.document_loader import DocumentProcessor
from module_04_retriever.retriever import MultiCollectionRetriever
from module_04_retriever.smart_router import SmartRouter
from module_05_generator.generator import Generator
from model.model_init import embeddings_model
from langgraph.graph import StateGraph, START, END


class RAGChain:
    """RAG链：整合组件"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.processor = DocumentProcessor(self.config)
        self.generator = Generator(self.config)
        self.embeddings = embeddings_model
        self.retriever = MultiCollectionRetriever(self.config)
        
        # 从配置中读取智能路由设置
        self.use_smart_routing = self.config.use_smart_routing
        
        # 初始化智能路由器
        if self.use_smart_routing:
            self.router = SmartRouter(self.config)
            print("✅ 智能路由已启用")
        else:
            self.router = None
            print("ℹ️  使用默认路由")
        
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建 LangGraph 流程"""

        def process_query(state: RAGState) -> RAGState:
            """处理查询：改写查询（如有对话历史）"""
            query = state["query"]
            chat_history = state.get("chat_history", [])

            if chat_history:
                rewritten = self.generator.rewrite_query(query, chat_history)
                print(f"🔄 查询改写：{query} -> {rewritten}")
                state["query"] = rewritten
            return state

        def retrieve_documents(state: RAGState) -> RAGState:
            """检索文档"""
            query = state["query"]
            
            # 智能路由：选择 collection
            if self.use_smart_routing and self.router:
                collections = self.router.route_query(query)
            else:
                collections = ["default"]
            
            # 从选中的 collection 检索文档
            all_docs = []
            for coll_key in collections:
                try:
                    docs = self.retriever.retrieve(query, self.embeddings, collection_key=coll_key)
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"⚠️  从集合 '{coll_key}' 检索失败: {e}")
            
            # 去重（基于内容）
            seen_contents = set()
            unique_docs = []
            for doc in all_docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    unique_docs.append(doc)
            
            docs = unique_docs
            print(f"📚 检索到 {len(docs)} 个相关文档")
            state["documents"] = docs

            context = []
            sources = []
            for i, doc in enumerate(docs):
                context.append(doc.page_content)
                sources.append({
                    "id": i,
                    "source": doc.metadata.get("source", "unknown"),
                    "content_preview": doc.page_content[:100] + "..."
                })
            state["context"] = "\n\n".join(context)
            state["sources"] = sources
            return state

        def generate_answer(state: RAGState) -> RAGState:
            answer = self.generator.generate(
                query=state["query"],
                context=state["context"],
                chat_history=state.get("chat_history", [])
            )
            state["answer"] = answer
            return state

        def evaluate_response(state: RAGState) -> RAGState:
            """评估回答置信度"""
            confidence = self.generator.evaluate_confidence(
                query=state["query"],
                context=state["context"],
                answer=state["answer"]
            )
            state["confidence"] = confidence
            print(f"📊 置信度评估：{confidence:.2f}")
            return state

        # 构建图
        graph = StateGraph(RAGState)

        graph.add_node("process_query", process_query)
        # 检索相关文档
        graph.add_node("retrieve", retrieve_documents)
        # 携带rag文档上下文，对话历史 生成回答
        graph.add_node("generate", generate_answer)
        # 根据rag文档上下问，评估回答置信度
        graph.add_node("evaluate", evaluate_response)

        graph.add_edge(START, "process_query")
        graph.add_edge("process_query", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "evaluate")
        graph.add_edge("evaluate", END)

        return graph.compile()

    def query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """执行查询"""
        print(f"🤔 用户查询：{query}")
        initial_state = {
            "query": query,
            "chat_history": chat_history or [],
            "documents": [],
            "context": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0
        }
        result = self.graph.invoke(initial_state)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"]
        }