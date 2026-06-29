"""
问答加载器
    支持多种格式：TXT、PDF、Markdown、CSV
    自动格式检测
    元数据提取
"""
from typing import List, Dict, Optional


from module_02_config.config_01_rag_config import RAGConfig
from module_01_embeddings.embedding import get_embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from milvus_sotre import SimpleMilvusStore,get_milvus_client


class DocumentProcessor:
    """文档处理器：加载、分块、向量化"""

    def __init__(self, config: RAGConfig):
        self.config = config  # 加载配置
        """
        通用文档 RecursiveCharacterTextSplitter ✅
        代码文件 CodeTextSplitter
        Markdown 文档 MarkdownHeaderTextSplitter
        严格 token 限制 TokenTextSplitter
        超快处理（不关心质量） CharacterTextSplitter
        """
        self.text_splitter = RecursiveCharacterTextSplitter(  # 创建文本分块器
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        self.embeddings = get_embeddings()
        self.vector_store = None

    # ============= 定义工具方法
    def load_documents(self, texts: List[str], source: Optional[List[Dict]] = None) -> List[Document]:
        """
        加载文档
            texts：文档
            source：文档来源（例：参考文献.doc）

        """
        documents = []
        for i, text in enumerate(texts):
            metadata = source[i] if source else {"source": f"doc_{i}"}
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """文档分块"""
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document], milvus_collection: str) -> SimpleMilvusStore:
        try:
            """转化为向量存储对象"""
            texts = [doc.page_content for doc in documents]
            vectors = self.embeddings.embed_documents(texts)
            dim = len(vectors[0])
            print(f"  向量维度: {dim}")

            client = get_milvus_client(self.config)
            # 创建集合（新版 quick setup 模式）
            if not client.has_collection(milvus_collection):
                client.create_collection(
                    collection_name=milvus_collection,
                    dimension=dim,
                    primary_field_name="id",
                    id_type="string",
                    vector_field_name="vector",
                    metric_type="COSINE",
                    auto_id=False,
                    max_length=65535,
                )
                print(f"  [OK] 集合 '{milvus_collection}' 已创建")
            # 插入数据
            data = [
                {"id": str(i), "vector": vec, "text": texts[i]}
                for i, vec in enumerate(vectors)
            ]
            client.insert(collection_name=milvus_collection, data=data)
            print(f"  [OK] 已插入 {len(data)} 条数据")
            return SimpleMilvusStore(client, milvus_collection, self.embeddings, self.config)

        except Exception as e:
            print(f"\n[错误] 索引失败: {e}")
            print("请确保 Milvus 服务正在运行")
            import traceback
            traceback.print_exc()
            return None

    def process(self, texts: List[str], metadatas: Optional[List[Dict]] = None, milvus_collection: str = "documents") -> SimpleMilvusStore:
        """完整处理流程：加载 -> 分块 -> 向量化"""
        print("📄 加载文档...")
        documents = self.load_documents(texts, metadatas)
        print(f"   加载了 {len(documents)} 个文档")

        print("✂️  分割文档...")
        chunks = self.split_documents(documents)
        print(f"   生成了 {len(chunks)} 个文本块")

        print("🔢 创建向量存储...")
        vector_store = self.create_vector_store(chunks, milvus_collection)
        print("   向量存储创建完成")

        return vector_store