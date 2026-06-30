from functools import lru_cache
from src.core.config import RAGConfig
from src.core.rag_chain import RAGChain


@lru_cache()
def get_config() -> RAGConfig:
    """获取 RAG 配置单例"""
    return RAGConfig()


_rag_chain: RAGChain | None = None


def get_rag_chain(config: RAGConfig = None) -> RAGChain:
    """获取 RAGChain 单例"""
    global _rag_chain
    if _rag_chain is None:
        cfg = config or get_config()
        _rag_chain = RAGChain(cfg)
    return _rag_chain
