from fastapi import FastAPI

from src.api.logging_setup import get_logger, setup_logging
from src.api.middleware import RequestLoggingMiddleware, global_exception_handler
from src.api.routes import documents, health, query, sessions
from src.api.tracing_setup import setup_tracing


def _eager_init():
    """启动时预热所有服务连接"""
    logger = get_logger("startup")
    from src.api.dependencies import get_config

    config = get_config()
    if not config.eager_init:
        logger.info("eager_init=False，跳过启动预热")
        return

    # 1. 预热 embedding 模型（模型加载仅首次，后续请求不触发）
    logger.info("加载 embedding 模型...")
    from src.core.embeddings import get_embeddings

    emb = get_embeddings()
    emb.embed_query("warmup")
    logger.info("embedding 模型就绪")

    # 2. 预热 Redis
    logger.info("连接 Redis: %s", config.redis_uri)
    from src.api.dependencies import get_session_store

    get_session_store(config=config)
    logger.info("Redis 就绪")

    # 3. 预热 Milvus（连接验证）
    logger.info("连接 Milvus: %s", config.milvus_uri)
    from src.core.milvus_store import get_milvus_client

    client = get_milvus_client(config)
    collections = client.list_collections()
    logger.info("Milvus 就绪, %d 个集合", len(collections))


def create_app() -> FastAPI:
    setup_logging()

    _eager_init()

    app = FastAPI(title="RAG System API", version="0.1.0")
    app.add_middleware(RequestLoggingMiddleware)
    app.add_exception_handler(Exception, global_exception_handler)

    setup_tracing(app)

    app.include_router(health.router)
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(sessions.router, prefix="/api/v1")

    return app


app = create_app()


def main():
    import uvicorn

    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
