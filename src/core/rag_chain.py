import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.api.logging_setup import get_logger
from src.api.metrics import rag_query_latency_seconds, rag_query_total, rag_retrieve_docs_count
from src.core.config import RAGConfig
from src.core.embeddings import get_embeddings
from src.core.generator import Generator
from src.core.retriever import MultiCollectionRetriever
from src.core.smart_router import SmartRouter
from src.core.state import RAGState


class RAGChain:
    """RAG 链：整合所有组件，构建 LangGraph 流水线"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.generator = Generator(self.config)
        self.embeddings = get_embeddings()
        self.retriever = MultiCollectionRetriever(self.config)
        self.use_smart_routing = self.config.use_smart_routing
        self.router = SmartRouter(self.config) if self.use_smart_routing else None
        self.graph = self._build_graph()

    def _build_graph(self):
        def process_query(state: RAGState) -> RAGState:
            query = state["query"]
            chat_history = state.get("chat_history", [])
            if chat_history:
                rewritten = self.generator.rewrite_query(query, chat_history)
                state["query"] = rewritten
            return state

        def retrieve_documents(state: RAGState) -> RAGState:
            query = state["query"]
            if self.use_smart_routing and self.router:
                collections = self.router.route_query(query)
            else:
                collections = ["default"]

            all_docs = []
            for coll_key in collections:
                try:
                    docs = self.retriever.retrieve(query, self.embeddings, collection_key=coll_key)
                    all_docs.extend(docs)
                except Exception as e:
                    get_logger("rag_chain").warning("检索集合失败", collection=coll_key, error=str(e))

            seen = set()
            unique_docs = []
            for doc in all_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)

            state["documents"] = unique_docs
            context_parts = []
            sources = []
            for i, doc in enumerate(unique_docs):
                context_parts.append(doc.page_content)
                sources.append(
                    {
                        "id": i,
                        "source": doc.metadata.get("source", "unknown"),
                        "content_preview": doc.page_content[:100] + "...",
                    }
                )
            state["context"] = "\n\n".join(context_parts)
            state["sources"] = sources
            return state

        def generate_answer(state: RAGState) -> RAGState:
            if state["context"]:
                state["answer"] = self.generator.generate(
                    query=state["query"], context=state["context"], chat_history=state.get("chat_history", [])
                )
            else:
                # 检索无结果时直接用大模型回答，不走 RAG 约束
                state["answer"] = self.generator.generate_chat(
                    query=state["query"], chat_history=state.get("chat_history", [])
                )
            return state

        def evaluate_response(state: RAGState) -> RAGState:
            state["confidence"] = self.generator.evaluate_confidence(
                query=state["query"], context=state["context"], answer=state["answer"]
            )
            return state

        graph = StateGraph(RAGState)
        graph.add_node("process_query", process_query)
        graph.add_node("retrieve", retrieve_documents)
        graph.add_node("generate", generate_answer)
        graph.add_node("evaluate", evaluate_response)
        graph.add_edge(START, "process_query")
        graph.add_edge("process_query", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "evaluate")
        graph.add_edge("evaluate", END)
        return graph.compile()

    def query(self, query: str, chat_history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        start = time.perf_counter()
        initial_state: RAGState = {
            "query": query,
            "chat_history": chat_history or [],
            "documents": [],
            "context": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0,
        }
        try:
            result = self.graph.invoke(initial_state)
            rag_query_total.labels(status="success").inc()
            rag_retrieve_docs_count.observe(len(result.get("documents", [])))
        except Exception:
            rag_query_total.labels(status="error").inc()
            raise
        finally:
            rag_query_latency_seconds.observe(time.perf_counter() - start)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
        }
