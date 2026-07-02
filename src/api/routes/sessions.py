from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_session_store
from src.core.session_store import SessionStore

router = APIRouter(tags=["sessions"])


class SessionSummary(BaseModel):
    id: str
    round_count: int
    created_at: str


class SessionsResponse(BaseModel):
    sessions: list[SessionSummary]


class RoundItem(BaseModel):
    role: str
    content: str
    timestamp: str


class SessionDetailResponse(BaseModel):
    rounds: list[RoundItem]


@router.get("/sessions", response_model=SessionsResponse)
def list_sessions(store: SessionStore = Depends(get_session_store)):
    sessions = store.list_sessions()
    return SessionsResponse(sessions=[SessionSummary(**s) for s in sessions])


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(session_id: str, store: SessionStore = Depends(get_session_store)):
    history = store.get_history(session_id)
    rounds = [RoundItem(**r) for r in history]
    return SessionDetailResponse(rounds=rounds)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, store: SessionStore = Depends(get_session_store)):
    store.delete_session(session_id)
    return {"deleted": True}
