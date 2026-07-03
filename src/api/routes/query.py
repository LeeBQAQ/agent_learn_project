import json as _json
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api.dependencies import get_rag_chain, get_session_store
from src.core.rag_chain import RAGChain
from src.core.session_store import SessionStore

router = APIRouter(tags=["query"])


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    chat_history: list[ChatMessage] | None = None
    top_k: int | None = None
    collections: list[str] | None = None
    session_id: str | None = None  # 新增


class SourceItem(BaseModel):
    source: str
    score: float
    content_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    confidence: float
    latency_ms: float
    session_id: str | None = None  # 新增


@router.post("/query", response_model=QueryResponse)
def rag_query(
    req: QueryRequest,
    rag: RAGChain = Depends(get_rag_chain),
    store: SessionStore = Depends(get_session_store),
):
    start = time.perf_counter()

    # 解析 session_id
    session_id = req.session_id or str(uuid.uuid4())

    # 从 SessionStore 获取历史
    chat_history = None
    if req.chat_history:
        chat_history = [{"role": m.role, "content": m.content} for m in req.chat_history]
    elif session_id:
        chat_history = store.get_history(session_id)

    result = rag.query(req.query, chat_history=chat_history)
    latency_ms = (time.perf_counter() - start) * 1000

    # 保存当前轮次到 SessionStore
    store.add_round(session_id, "user", req.query)
    store.add_round(session_id, "assistant", result.get("answer", ""))

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
        session_id=session_id,
    )


@router.post("/query/stream")
async def rag_query_stream(
    req: QueryRequest,
    rag: RAGChain = Depends(get_rag_chain),
    store: SessionStore = Depends(get_session_store),
):
    session_id = req.session_id or str(uuid.uuid4())

    chat_history = None
    if req.chat_history:
        chat_history = [{"role": m.role, "content": m.content} for m in req.chat_history]
    elif session_id:
        chat_history = store.get_history(session_id)

    def generate():
        full_answer = ""
        for event_type, data in rag.stream_query(req.query, chat_history=chat_history):
            yield f"event: {event_type}\ndata: {data}\n\n"
            if event_type == "token":
                content = _json.loads(data).get("content", "")
                full_answer += content
            elif event_type == "error":
                break

        if full_answer:
            store.add_round(session_id, "user", req.query)
            store.add_round(session_id, "assistant", full_answer)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
