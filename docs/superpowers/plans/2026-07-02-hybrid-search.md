# 混合检索（BM25 + 向量 + RRF）— 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 RAG 系统添加 BM25 关键词 + 向量语义的混合检索，通过 RRF 融合两路结果提升召回率。

**Architecture:** `RAGConfig` 新增 `hybrid_search` 开关控制全局行为。`SimpleMilvusStore` 新增 `bm25_search()` 和 `hybrid_search()` 方法，后者并行执行向量和 BM25 检索后用 RRF (k=60) 融合。`MultiCollectionRetriever.retrieve()` 根据 config 自动路由到纯向量或混合检索。API 不变，对调用方透明。

**Tech Stack:** Python 3.12, Milvus 2.4+ (内置 BM25), pymilvus

## Global Constraints

- 默认 `hybrid_search = False`，向后兼容
- RRF 参数 k = 60
- 两路检索各取 `top_k * 2` 再做融合
- 已有 collection 无 BM25 索引时降级为纯向量检索
- API 请求/响应格式不变

---

### Task 1: RAGConfig 新增 hybrid_search 开关

**Files:**
- Modify: `src/core/config.py:24-44`
- Test: `tests/unit/test_config.py` (new)

**Interfaces:**
- Produces: `RAGConfig.hybrid_search: bool = False`

- [ ] **Step 1: 写测试**

```python
# tests/unit/test_config.py
from src.core.config import RAGConfig


def test_hybrid_search_default_false():
    config = RAGConfig()
    assert config.hybrid_search is False


def test_hybrid_search_can_enable():
    config = RAGConfig(hybrid_search=True)
    assert config.hybrid_search is True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL — `unexpected keyword argument 'hybrid_search'`

- [ ] **Step 3: 在 `src/core/config.py` 的 RAGConfig 中添加字段**

在 `max_tokens: int = 1024` 之后添加：

```python
    # 混合检索配置
    hybrid_search: bool = False
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/test_config.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/core/config.py tests/unit/test_config.py
git commit -m "feat: RAGConfig 新增 hybrid_search 开关"
```

---

### Task 2: SimpleMilvusStore 新增 bm25_search + hybrid_search

**Files:**
- Modify: `src/core/milvus_store.py:1-50` (新增方法)
- Test: `tests/unit/test_hybrid_search.py` (new)

**Interfaces:**
- Produces:
  - `SimpleMilvusStore.bm25_search(query: str, k: int) -> List[Document]` — BM25 关键词检索
  - `SimpleMilvusStore.hybrid_search(query: str, k: int) -> List[Document]` — RRF 融合检索
- Consumes: `RAGConfig.hybrid_search`, `MilvusClient.query()` with `TEXT_MATCH`

- [ ] **Step 1: 写单测**

```python
# tests/unit/test_hybrid_search.py
from unittest.mock import MagicMock
from src.core.milvus_store import SimpleMilvusStore
from src.core.config import RAGConfig
from langchain_core.documents import Document


def test_bm25_search_basic():
    config = RAGConfig(hybrid_search=True, top_k=2)
    mock_client = MagicMock()
    mock_client.query.return_value = [
        {"id": "0", "text": "Python GIL 全局解释器锁", "source": "doc1.txt"},
        {"id": "1", "text": "Python 多线程编程", "source": "doc2.txt"},
    ]
    mock_embeddings = MagicMock()

    store = SimpleMilvusStore(mock_client, "test_coll", mock_embeddings, config)
    docs = store.bm25_search("GIL", k=3)

    assert len(docs) == 2
    assert docs[0].page_content == "Python GIL 全局解释器锁"
    mock_client.query.assert_called_once()
    call_args = mock_client.query.call_args[1]
    assert "TEXT_MATCH" in call_args["filter"]


def test_bm25_search_empty_result():
    config = RAGConfig(hybrid_search=True)
    mock_client = MagicMock()
    mock_client.query.return_value = []
    mock_embeddings = MagicMock()

    store = SimpleMilvusStore(mock_client, "test_coll", mock_embeddings, config)
    docs = store.bm25_search("nonexistent")

    assert docs == []


def test_hybrid_search_rrf_merge():
    config = RAGConfig(hybrid_search=True, top_k=2)
    mock_client = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 384

    # 向量检索返回：doc_A(rank1), doc_B(rank2)
    mock_client.search.return_value = [[
        {"entity": {"text": "Doc A content", "source": "a.txt"}, "distance": 0.95},
        {"entity": {"text": "Doc B content", "source": "b.txt"}, "distance": 0.80},
    ]]
    # BM25 检索返回：doc_B(rank1), doc_A(rank2)
    mock_client.query.return_value = [
        {"id": "1", "text": "Doc B content", "source": "b.txt"},
        {"id": "0", "text": "Doc A content", "source": "a.txt"},
    ]

    store = SimpleMilvusStore(mock_client, "test_coll", mock_embeddings, config)
    docs = store.hybrid_search("test query", k=2)

    assert len(docs) <= 2
    # Both doc_A and doc_B should appear (RRF merges different rankings)
    contents = [d.page_content for d in docs]
    assert "Doc A content" in contents
    assert "Doc B content" in contents


def test_rrf_formula_correctness():
    """验证 RRF 公式: score = sum 1/(k + rank) for k=60"""
    from src.core.milvus_store import _rrf_fusion

    vec_docs = [
        Document(page_content="A", metadata={"source": "a.txt", "score": 0.95}),
        Document(page_content="B", metadata={"source": "b.txt", "score": 0.80}),
    ]
    bm25_docs = [
        Document(page_content="B", metadata={"source": "b.txt", "score": 0.0}),
        Document(page_content="A", metadata={"source": "a.txt", "score": 0.0}),
    ]

    result = _rrf_fusion(vec_docs, bm25_docs, k=60, top_k=2)
    assert len(result) == 2
    # B ranks (2 in vec, 1 in bm25) -> 1/(60+2) + 1/(60+1) ≈ 0.0161 + 0.0164 ≈ 0.0325
    # A ranks (1 in vec, 2 in bm25) -> 1/(60+1) + 1/(60+2) ≈ 0.0164 + 0.0161 ≈ 0.0325
    # Equal scores, so order depends on sort stability
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/test_hybrid_search.py -v`
Expected: FAIL — `AttributeError: 'SimpleMilvusStore' object has no attribute 'bm25_search'`

- [ ] **Step 3: 在 `src/core/milvus_store.py` 中实现代码**

在 `SimpleMilvusStore` 类的 `as_retriever` 方法之后添加：

```python
    def bm25_search(self, query: str, k: int = 3) -> List[Document]:
        """BM25 关键词检索（Milvus TEXT_MATCH）"""
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter=f'TEXT_MATCH("text", "{query}")',
                output_fields=["text", "source"],
                limit=k,
            )
            docs = []
            for entity in results:
                docs.append(Document(
                    page_content=entity.get("text", ""),
                    metadata={"source": entity.get("source", ""), "score": 0.0},
                ))
            return docs
        except Exception as e:
            print(f"Warning: BM25 检索失败: {e}")
            return []

    def hybrid_search(self, query: str, k: int = 3) -> List[Document]:
        """混合检索：向量 + BM25 + RRF 融合"""
        fetch_k = k * 2

        vector_docs = self.similarity_search(query, k=fetch_k)
        bm25_docs = self.bm25_search(query, k=fetch_k)

        return _rrf_fusion(vector_docs, bm25_docs, k=60, top_k=k)
```

在文件末尾（`get_milvus_client` 函数之前）添加 RRF 融合函数：

```python
def _rrf_fusion(
    vector_docs: List[Document],
    bm25_docs: List[Document],
    k: int = 60,
    top_k: int = 3,
) -> List[Document]:
    """RRF (Reciprocal Rank Fusion) 融合两路检索结果"""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(vector_docs, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in doc_map:
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_k]]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/test_hybrid_search.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/core/milvus_store.py tests/unit/test_hybrid_search.py
git commit -m "feat: SimpleMilvusStore 新增 bm25_search + hybrid_search (RRF 融合)"
```

---

### Task 3: DocumentProcessor 创建 collection 时启用 BM25

**Files:**
- Modify: `src/core/document_loader.py:49-71` (`create_vector_store` 方法)

**Interfaces:**
- Consumes: `RAGConfig.hybrid_search`
- Produces: Milvus collection 带 BM25 Function 注册

- [ ] **Step 1: 修改 `create_vector_store` 中的 collection 创建逻辑**

在 `src/core/document_loader.py` 的 `create_vector_store` 方法中，将 `client.create_collection(...)` 调用改为：

```python
        client = get_milvus_client(self.config)
        if not client.has_collection(milvus_collection):
            create_params = dict(
                collection_name=milvus_collection,
                dimension=dim,
                primary_field_name="id",
                id_type="string",
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False,
                max_length=65535,
            )
            if self.config.hybrid_search:
                create_params["enable_dynamic_field"] = True
            client.create_collection(**create_params)
```

- [ ] **Step 2: 验证导入和语法正确**

Run: `python -c "from src.core.document_loader import DocumentProcessor; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add src/core/document_loader.py
git commit -m "feat: DocumentProcessor 创建 collection 时支持 BM25 (enable_dynamic_field)"
```

---

### Task 4: MultiCollectionRetriever 路由到混合检索

**Files:**
- Modify: `src/core/retriever.py:26-34` (`retrieve` 方法)

**Interfaces:**
- Consumes: `SimpleMilvusStore.hybrid_search()`, `RAGConfig.hybrid_search`
- Produces: `MultiCollectionRetriever.retrieve()` 根据 config 自动选择检索方式

- [ ] **Step 1: 修改 `retrieve` 方法**

将 `src/core/retriever.py` 的 `MultiCollectionRetriever.retrieve` 方法中的检索调用改为：

```python
    def retrieve(
        self, query: str, embeddings, collection_key: str = "default", k: Optional[int] = None
    ) -> List[Document]:
        if collection_key not in self.config.collections:
            collection_key = "default"
        store = self._get_store(collection_key, embeddings)
        coll_config = self.config.get_collection_config(collection_key)
        k = k or coll_config.top_k

        if self.config.hybrid_search:
            return store.hybrid_search(query=query, k=k)
        return store.similarity_search(query=query, k=k)
```

- [ ] **Step 2: 运行已有测试确认不破坏**

Run: `pytest tests/unit/ tests/api/ -v`
Expected: All existing tests pass

- [ ] **Step 3: Commit**

```bash
git add src/core/retriever.py
git commit -m "feat: MultiCollectionRetriever 支持混合检索路由"
```

---

## 验证方式

```bash
# 1. 运行全部测试
pytest tests/ -v

# 2. 验证混合检索开关
python -c "
from src.core.config import RAGConfig
c = RAGConfig()
print('hybrid_search default:', c.hybrid_search)  # False
c2 = RAGConfig(hybrid_search=True)
print('hybrid_search on:', c2.hybrid_search)       # True
"

# 3. 验证 RRF 公式
python -c "
from src.core.milvus_store import _rrf_fusion
from langchain_core.documents import Document
d1 = Document(page_content='A', metadata={'source': 'a.txt', 'score': 0.95})
d2 = Document(page_content='B', metadata={'source': 'b.txt', 'score': 0.80})
result = _rrf_fusion([d1, d2], [d2, d1], k=60, top_k=2)
print('RRF result:', [d.page_content for d in result])
"

# 4. 如有 Milvus 运行，可上传文档后查询验证
```
