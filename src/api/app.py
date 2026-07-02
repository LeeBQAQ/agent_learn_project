from fastapi import FastAPI

from src.api.middleware import RequestLoggingMiddleware, global_exception_handler
from src.api.routes import documents, health, query
from src.api.tracing_setup import setup_tracing


def create_app() -> FastAPI:
    app = FastAPI(title="RAG System API", version="0.1.0")
    app.add_middleware(RequestLoggingMiddleware)
    app.add_exception_handler(Exception, global_exception_handler)

    setup_tracing(app)

    app.include_router(health.router)
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")

    return app


app = create_app()


def main():
    import uvicorn

    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
