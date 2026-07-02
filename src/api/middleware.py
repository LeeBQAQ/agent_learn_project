import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.logging_setup import get_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件：记录 method、path、status、耗时"""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        logger = get_logger("api")
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(elapsed_ms, 1),
        )
        return response


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger = get_logger("api")
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error("unhandled_exception", request_id=request_id, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )
