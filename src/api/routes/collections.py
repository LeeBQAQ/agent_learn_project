from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_config
from src.core.config import RAGConfig, sanitize_collection_name
from src.core.milvus_store import get_milvus_client

router = APIRouter(tags=["collections"])


class CollectionInfo(BaseModel):
    key: str
    name: str
    description: str
    top_k: int


class CreateCollectionRequest(BaseModel):
    key: str
    description: str = ""


@router.get("/collections", response_model=list[CollectionInfo])
def list_collections(config: RAGConfig = Depends(get_config)):
    client = get_milvus_client(config)
    config.sync_collections(client)
    return [
        CollectionInfo(key=k, name=c.name, description=c.description, top_k=c.top_k)
        for k, c in config.collections.items()
    ]


@router.post("/collections/create")
def create_collection(req: CreateCollectionRequest, config: RAGConfig = Depends(get_config)):
    key = sanitize_collection_name(req.key)
    if key == "default":
        return {"error": "default 为保留 key"}
    milvus_name = f"rag_{key}"
    client = get_milvus_client(config)
    if not client.has_collection(milvus_name):
        client.create_collection(
            collection_name=milvus_name,
            dimension=384,
            primary_field_name="id",
            id_type="string",
            vector_field_name="vector",
            metric_type="COSINE",
            auto_id=False,
            max_length=65535,
            enable_dynamic_field=True,
        )
    config.register_collection(key, milvus_name, description=req.description)
    return {"key": key, "name": milvus_name, "status": "created"}


@router.delete("/collections/{key}")
def delete_collection(key: str, config: RAGConfig = Depends(get_config)):
    if key not in config.collections:
        return {"error": "集合不存在"}
    coll_name = config.collections[key].name
    client = get_milvus_client(config)
    if client.has_collection(coll_name):
        client.drop_collection(coll_name)
    del config.collections[key]
    return {"key": key, "deleted": True}
