# 混合检索（BM25 + 向量 + RRF）— 设计文档

## Context

当前 RAG 系统仅使用余弦相似度的向量检索。对于精确术语匹配（人名、API 名称、错误码）效果差。混合检索融合 BM25 关键词匹配和语义向量检索，通过 RRF 合并结果，提升召回率。

## 改动范围

| 文件 | 改动 |
|------|------|
| `src/core/config.py` | RAGConfig 新增 `hybrid_search: bool = False` |
| `src/core/document_loader.py` | 创建 collection 时注册 BM25 Function |
| `src/core/milvus_store.py` | 新增 `bm25_search()` + `hybrid_search()` 方法 |
| `src/core/retriever.py` | `retrieve()` 根据 config 选纯向量/混合检索 |

## 核心流程

```
query
  │
  ▼
SimpleMilvusStore.hybrid_search()
  │
  ├─ 向量检索 (COSINE) ──→ top_k * 2 docs
  ├─ BM25 检索 (TEXT_MATCH) ──→ top_k * 2 docs
  │
  └─ RRF 融合 ──→ top_k docs
```

## RRF 公式

```
RRF_score(doc) = Σ 1 / (k + rank_i)    i∈{vector, bm25}
k = 60
```

## 索引阶段

`DocumentProcessor.create_vector_store()` 创建 collection 时：
- 调用 `client.create_collection(..., enable_dynamic_field=True)`
- 使用 Milvus 内置 analyzer 注册 BM25 function

## 检索阶段

`SimpleMilvusStore` 新增两个方法：

- `bm25_search(query, k)` — 用 `client.query()` + `TEXT_MATCH` 表达式进行关键词检索
- `hybrid_search(query, k)` — 并行执行向量检索 (top_k*2) 和 BM25 检索 (top_k*2)，RRF 融合返回 top_k

## 兼容性

- `hybrid_search = False`（默认）→ 行为完全不变
- 已有 collection 无 BM25 索引 → 降级为纯向量检索

## API

请求/响应不变，检索方式由服务端配置控制。
