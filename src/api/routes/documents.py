import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends
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


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection: Optional[str] = Form(None),
    config: RAGConfig = Depends(get_config),
):
    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    doc_id = str(uuid.uuid4())[:8]

    processor = DocumentProcessor(config)
    docs = processor.load_documents([text], [{"source": file.filename, "id": doc_id}])
    classified = processor.classify_and_store(docs, collection or "default")

    chunk_count = 0
    target_collection = collection or "default"
    for coll_key, doc_list in classified.items():
        chunks = processor.split_documents(doc_list)
        chunk_count = len(chunks)
        coll_config = config.get_collection_config(coll_key)
        processor.create_vector_store(chunks, coll_config.name)
        target_collection = coll_key

    return UploadResponse(
        document_id=doc_id,
        filename=file.filename or "unknown",
        collection=target_collection,
        chunk_count=chunk_count,
        status="indexed",
    )


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
            client.delete(coll_config.name, filter=f'source like "%{document_id}%"')
            deleted = True
        except Exception:
            pass
    return {"document_id": document_id, "deleted": deleted}
