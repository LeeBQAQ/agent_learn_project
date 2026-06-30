import time
import uuid
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件：记录 method、path、status、耗时"""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        print(
            f"request_id={request_id} method={request.method} path={request.url.path} "
            f"status={response.status_code} latency_ms={elapsed_ms:.1f}"
        )
        return response


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """全局异常处理，统一返回 JSON 错误格式"""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)}
    )
