# RAG System

企业级 RAG（Retrieval-Augmented Generation）检索增强生成系统。

- **LLM**: DeepSeek (via langchain-deepseek)
- **向量数据库**: Milvus 2.4+
- **Embedding**: HuggingFace (sentence-transformers) / OpenAI 可选
- **检索**: 向量语义检索 + BM25 关键词检索 + RRF 混合融合
- **API**: FastAPI RESTful
- **可观测性**: structlog / Prometheus / OpenTelemetry
- **部署**: Docker Compose / Kubernetes

---

## 快速开始

### 环境要求

- Python 3.12+
- Milvus 2.4+（本地或 Docker）
- DeepSeek API Key

### 安装

```bash
git clone <repo-url>
cd agent_learn_project

# 安装依赖
pip install -e .

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY 和 MODEL
```

### 启动 API 服务

```bash
python -m src.api.app
# FastAPI 启动在 http://127.0.0.1:8000
# API 文档: http://127.0.0.1:8000/docs
```

### Docker Compose 一键启动

```bash
docker compose up -d
# 包含: Milvus + FastAPI + Prometheus + Loki + Tempo + Grafana
```

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查（含 Milvus 连通性） |
| `GET` | `/metrics` | Prometheus 指标 |
| `POST` | `/api/v1/query` | RAG 问答 |
| `POST` | `/api/v1/documents/upload` | 批量上传文档 |
| `GET` | `/api/v1/documents` | 列出已索引文档 |
| `DELETE` | `/api/v1/documents/{id}` | 删除文档 |

### 示例

```bash
# 问答
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是 RAG？"}'

# 上传文档
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "files=@doc.pdf" -F "collection=ai_ml"
```

---

## 项目结构

```
├── src/
│   ├── api/                    # FastAPI 层
│   │   ├── app.py              # 应用工厂
│   │   ├── dependencies.py     # 依赖注入
│   │   ├── middleware.py       # 请求日志 / 异常处理
│   │   ├── metrics.py          # Prometheus 指标
│   │   ├── logging_setup.py    # structlog 结构化日志
│   │   ├── tracing_setup.py    # OpenTelemetry 链路追踪
│   │   └── routes/             # 路由: health / query / documents
│   ├── core/                   # RAG 核心逻辑
│   │   ├── config.py           # RAGConfig + SmartMode
│   │   ├── state.py            # RAGState (LangGraph)
│   │   ├── model.py            # LLM 初始化
│   │   ├── embeddings.py       # Embedding 模型选择
│   │   ├── rag_chain.py        # LangGraph 流水线编排
│   │   ├── document_loader.py  # 文档加载/分块/分类/存储
│   │   ├── milvus_store.py     # Milvus 存储 + BM25 + RRF
│   │   ├── retriever.py        # 多 Collection 检索器
│   │   ├── smart_router.py     # LLM 智能路由/分类
│   │   ├── generator.py        # 回答生成/改写/评估
│   │   └── templates.py        # Prompt 模板库
│   └── cli/                    # CLI 交互入口
├── tests/
│   ├── unit/                   # 单测 (5 文件)
│   ├── integration/            # 集成测试
│   └── api/                    # API 测试
├── deploy/
│   ├── k8s/                    # K8s 部署资源
│   ├── grafana/                # Grafana 数据源
│   ├── prometheus.yml          # Prometheus 配置
│   └── tempo.yaml              # Tempo 配置
├── legacy/                     # 原始学习代码（只读）
├── Dockerfile                  # 多阶段构建
├── docker-compose.yaml         # 全栈本地环境
├── pyproject.toml              # 依赖 + lint/mypy/pytest 配置
└── .github/workflows/ci.yml   # CI/CD 流水线
```

---

## 架构

```
请求 → FastAPI (def endpoint, 线程池)
         │
         └→ RAGChain (LangGraph 流水线)
              ├─ process_query      (查询改写)
              ├─ retrieve           (混合检索)
              │    ├─ SmartRouter   (LLM 选择 Collection)
              │    └─ MultiCollectionRetriever
              │         └─ SimpleMilvusStore
              │              ├─ similarity_search  (向量 COSINE)
              │              └─ bm25_search        (BM25 关键词)
              │                   └─ RRF 融合 (k=60)
              ├─ generate           (LLM 生成回答)
              └─ evaluate           (置信度评估)
```

### 混合检索

`RAGConfig(hybrid_search=True)` 启用 BM25 + 向量 + RRF 混合检索：

```python
config = RAGConfig(hybrid_search=True)
```

默认关闭，向后兼容纯向量检索。

### 智能路由

`SmartMode` 控制 LLM 参与的智能功能：

- `DISABLED` — 禁用，所有文档进 `default` 集合
- `ROUTING_ONLY` — LLM 选择查询的 Collection
- `FULL` — LLM 分类文档 + 路由查询（默认）

---

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# 类型检查
mypy src/

# 测试
pytest tests/ -v
```

---

## K8s 部署

```bash
kubectl apply -k deploy/k8s/
```

---

## 可观测性

| 组件 | 端口 | 用途 |
|------|------|------|
| Prometheus | `:9090` | 指标采集，抓取 `/metrics` |
| Loki | `:3100` | 日志聚合，采集 JSON 日志 |
| Tempo | `:4317` | 链路追踪，OTLP gRPC |
| Grafana | `:3000` | 统一可视化面板 |

---

## CI/CD

GitHub Actions 流水线：

`PR` → lint → type-check → test → build image  
`main` → ... → deploy staging  
`tag v*` → ... → deploy prod
