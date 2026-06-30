import time
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.api.dependencies import get_config
from src.core.config import RAGConfig
from src.core.milvus_store import get_milvus_client

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    milvus: str
    model: str
    timestamp: float


@router.get("/health", response_model=HealthResponse)
def health_check(config: RAGConfig = Depends(get_config)):
    milvus_status = "disconnected"
    try:
        client = get_milvus_client(config)
        client.list_collections()
        milvus_status = "connected"
    except Exception:
        pass

    actual_status = "healthy" if milvus_status == "connected" else "degraded"
    return HealthResponse(
        status=actual_status,
        milvus=milvus_status,
        model="ready",
        timestamp=time.time(),
    )
