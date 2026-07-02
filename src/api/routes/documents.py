import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from pydantic import BaseModel

from src.api.dependencies import get_config
from src.core.config import RAGConfig
from src.core.document_loader import DocumentProcessor
from src.core.milvus_store import get_milvus_client

router = APIRouter(tags=["documents"])


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    collection: str
    chunk_count: int
    status: str


class BatchUploadResponse(BaseModel):
    total: int
    results: list[UploadResponse]


def _process_document(text: str, filename: str, doc_id: str, collection: str | None, config: RAGConfig) -> None:
    """后台处理：分类 → 分块 → 向量化 → 存入 Milvus"""
    processor = DocumentProcessor(config)

    docs = processor.load_documents([text], [{"source": filename, "id": doc_id}])
    classified = processor.classify_and_store(docs, collection or "default")

    for coll_key, doc_list in classified.items():
        chunks = processor.split_documents(doc_list)
        coll_config = config.get_collection_config(coll_key)
        processor.create_vector_store(chunks, coll_config.name)


@router.post("/documents/upload", response_model=BatchUploadResponse)
def upload_document(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    collection: str | None = Form(None),
    config: RAGConfig = Depends(get_config),
):
    results: list[UploadResponse] = []

    for file in files:
        content = file.file.read()
        text = content.decode("utf-8", errors="replace")
        doc_id = str(uuid.uuid4())[:8]

        background_tasks.add_task(_process_document, text, file.filename or "unknown", doc_id, collection, config)

        results.append(
            UploadResponse(
                document_id=doc_id,
                filename=file.filename or "unknown",
                collection=collection or "default",
                chunk_count=0,
                status="processing",
            )
        )

    return BatchUploadResponse(total=len(results), results=results)


@router.get("/documents")
def list_documents(config: RAGConfig = Depends(get_config)):
    client = get_milvus_client(config)
    result = {}
    for key, coll_config in config.collections.items():
        try:
            if client.has_collection(coll_config.name):
                stats = client.get_collection_stats(coll_config.name)
                result[key] = {"collection_name": coll_config.name, "row_count": stats.get("row_count", 0)}
        except Exception:
            result[key] = {"collection_name": coll_config.name, "row_count": "unknown"}
    return {"collections": result}


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, config: RAGConfig = Depends(get_config)):
    client = get_milvus_client(config)
    deleted = False
    for coll_config in config.collections.values():
        try:
            if client.has_collection(coll_config.name):
                client.delete(coll_config.name, filter=f'source like "%{document_id}%"')
                deleted = True
        except Exception:
            pass
    return {"document_id": document_id, "deleted": deleted}
