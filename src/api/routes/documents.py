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
        processor.create_vector_store(chunks, coll_config.name, doc_id=doc_id)


def _replace_document(text: str, filename: str, doc_id: str, collection: str | None, config: RAGConfig) -> None:
    """后台替换：先删旧 chunk，再插入新 chunk"""
    client = get_milvus_client(config)
    for _, coll_config in config.collections.items():
        try:
            if client.has_collection(coll_config.name):
                client.delete(coll_config.name, filter=f'doc_id == "{doc_id}"')
        except Exception:
            pass
    _process_document(text, filename, doc_id, collection, config)


@router.post("/documents/upload", response_model=BatchUploadResponse)
def upload_document(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    collection: str | None = Form(None),
    doc_id: str | None = Form(None),
    config: RAGConfig = Depends(get_config),
):
    results: list[UploadResponse] = []

    for file in files:
        content = file.file.read()
        text = content.decode("utf-8", errors="replace")
        did = doc_id or str(uuid.uuid4())[:8]

        if doc_id:
            background_tasks.add_task(_replace_document, text, file.filename or "unknown", did, collection, config)
            status = "replacing"
        else:
            background_tasks.add_task(_process_document, text, file.filename or "unknown", did, collection, config)
            status = "processing"

        results.append(UploadResponse(
            document_id=did,
            filename=file.filename or "unknown",
            collection=collection or "default",
            chunk_count=0,
            status=status,
        ))

    return BatchUploadResponse(total=len(results), results=results)


@router.get("/documents")
def list_documents(config: RAGConfig = Depends(get_config)):
    client = get_milvus_client(config)
    docs: dict[str, dict] = {}

    for key, coll_config in config.collections.items():
        try:
            if not client.has_collection(coll_config.name):
                continue
            results = client.query(
                collection_name=coll_config.name,
                filter="doc_id != ''",
                output_fields=["doc_id", "source", "chunk_index"],
                limit=10000,
            )
            for row in results:
                did = row.get("doc_id", "")
                if not did:
                    continue
                if did not in docs:
                    docs[did] = {"doc_id": did, "filename": row.get("source", "unknown"), "chunk_count": 0, "collections": []}
                docs[did]["chunk_count"] += 1
                if key not in docs[did]["collections"]:
                    docs[did]["collections"].append(key)
        except Exception:
            pass

    return {"documents": list(docs.values())}


@router.get("/documents/{document_id}")
def get_document(document_id: str, config: RAGConfig = Depends(get_config)):
    client = get_milvus_client(config)
    chunks = []
    filename = ""
    coll_name = ""

    for key, coll_config in config.collections.items():
        try:
            if not client.has_collection(coll_config.name):
                continue
            results = client.query(
                collection_name=coll_config.name,
                filter=f'doc_id == "{document_id}"',
                output_fields=["source", "chunk_index", "text"],
                limit=10000,
            )
            for row in results:
                if not filename:
                    filename = row.get("source", "")
                    coll_name = key
                chunks.append({
                    "chunk_index": row.get("chunk_index", 0),
                    "content_preview": row.get("text", "")[:100] + "...",
                })
        except Exception:
            pass

    chunks.sort(key=lambda c: c["chunk_index"])
    return {
        "doc_id": document_id,
        "filename": filename,
        "collection": coll_name,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, config: RAGConfig = Depends(get_config)):
    client = get_milvus_client(config)
    total_removed = 0
    for coll_config in config.collections.values():
        try:
            if client.has_collection(coll_config.name):
                result = client.delete(coll_config.name, filter=f'doc_id == "{document_id}"')
                total_removed += result.get("delete_count", 0)
        except Exception:
            pass
    return {"document_id": document_id, "deleted": total_removed > 0, "chunks_removed": total_removed}
