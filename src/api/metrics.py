from fastapi import Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest

rag_query_total = Counter("rag_query_total", "Total RAG queries", ["status"])
rag_query_latency_seconds = Histogram(
    "rag_query_latency_seconds", "Query latency in seconds", buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)
rag_retrieve_docs_count = Histogram(
    "rag_retrieve_docs_count", "Documents retrieved per query", buckets=[0, 1, 3, 5, 10, 20]
)
milvus_connection_status = Gauge("milvus_connection_status", "Milvus connection status (1=connected, 0=disconnected)")


def metrics_endpoint() -> Response:
    return Response(content=generate_latest(), media_type="text/plain")
