from functools import lru_cache

from fastapi import Depends

from src.core.config import RAGConfig
from src.core.rag_chain import RAGChain
from src.core.session_store import SessionStore


@lru_cache
def get_config() -> RAGConfig:
    """获取 RAG 配置单例"""
    return RAGConfig()


_rag_chain: RAGChain | None = None


def get_rag_chain(config: RAGConfig = Depends(get_config)) -> RAGChain:
    """获取 RAGChain 单例"""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain(config)
    return _rag_chain


_session_store: SessionStore | None = None


def get_session_store(config: RAGConfig = Depends(get_config)) -> SessionStore:
    """获取 SessionStore 单例"""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore(config)
    return _session_store
