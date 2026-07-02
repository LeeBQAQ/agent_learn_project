import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_rag_chain
from src.core.rag_chain import RAGChain

router = APIRouter(tags=["query"])


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    chat_history: list[ChatMessage] | None = None
    top_k: int | None = None
    collections: list[str] | None = None


class SourceItem(BaseModel):
    source: str
    score: float
    content_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    confidence: float
    latency_ms: float


@router.post("/query", response_model=QueryResponse)
def rag_query(req: QueryRequest, rag: RAGChain = Depends(get_rag_chain)):
    start = time.perf_counter()
    chat_history = None
    if req.chat_history:
        chat_history = [{"role": m.role, "content": m.content} for m in req.chat_history]
    result = rag.query(req.query, chat_history=chat_history)
    latency_ms = (time.perf_counter() - start) * 1000

    sources = [
        SourceItem(
            source=s.get("source", "unknown"), score=s.get("score", 0.0), content_preview=s.get("content_preview", "")
        )
        for s in result.get("sources", [])
    ]

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
        confidence=result.get("confidence", 0.0),
        latency_ms=round(latency_ms, 1),
    )
