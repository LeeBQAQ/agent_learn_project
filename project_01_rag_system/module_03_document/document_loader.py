"""
问答加载器
    支持多种格式：TXT、PDF、Markdown、CSV
    自动格式检测
    元数据提取
"""
from typing import List, Dict, Optional

from model.model_init import embeddings_model
from module_02_config.config_01_rag_config import RAGConfig
from module_04_retriever.smart_router import DocumentClassifier
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from module_03_document.milvus_store import SimpleMilvusStore, get_milvus_client


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
        self.embeddings = embeddings_model
        self.vector_store = None

        # 从配置中读取智能分类设置
        self.use_smart_classification = config.use_smart_classification
        if self.use_smart_classification:
            self.classifier = DocumentClassifier(config)
            print("✅ 智能文档分类已启用")
        else:
            self.classifier = None
            print("ℹ️  使用默认文档分类")

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

    def classify_and_store(
        self,
        documents: List[Document],
        default_collection: str = "default"
   ) -> Dict[str, List[Document]]:
        """智能分类文档并分组
        
        Args:
            documents: 文档列表
            default_collection: 默认集合名称
            
        Returns:
            {collection_name: [documents]} 字典
        """
        classified_docs = {}

        for doc in documents:
            if self.use_smart_classification and self.classifier:
                # 使用 LLM 分类
                result = self.classifier.classify_document(doc.page_content)
                collection = result.get("collection", default_collection)
                confidence = result.get("confidence", 0.5)

                # 验证 collection 是否存在
                if collection not in self.config.collections:
                    print(f"⚠️  集合 '{collection}' 不存在，使用默认集合")
                    collection = default_collection

                print(f"📄 文档分类: {collection} (置信度: {confidence:.2f})")
            else:
                collection = default_collection

            if collection not in classified_docs:
                classified_docs[collection] = []
            classified_docs[collection].append(doc)

        return classified_docs

    def create_vector_store(self, documents: List[Document], milvus_collection: str) -> SimpleMilvusStore:
        try:
            """转化为向量存储对象"""
            texts = [doc.page_content for doc in documents]
            sources = [doc.metadata.get("source", "") for doc in documents]  # 提取 source
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
                {"id": str(i), "vector": vec, "text": texts[i],"source": sources[i]}
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

    def process(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        milvus_collection: str = "documents",
        use_smart_classification: bool = None
    ) -> Dict[str, SimpleMilvusStore]:
        """完整处理流程：加载 -> 分类 -> 分块 -> 向量化
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            milvus_collection: 默认集合名称
            use_smart_classification: 是否使用智能分类（覆盖构造函数设置）
            
        Returns:
            {collection_name: vector_store} 字典
        """
        # 允许临时覆盖智能分类设置
        if use_smart_classification is not None:
            original_setting = self.use_smart_classification
            self.use_smart_classification = use_smart_classification

        print("📄 加载文档...")
        documents = self.load_documents(texts, metadatas)
        print(f"   加载了 {len(documents)} 个文档")

        print("🎯 智能分类文档...")
        # 先对完整文档进行分类
        classified_docs = self.classify_and_store(documents, milvus_collection)
        
        print(f"   文档分类完成，共 {len(classified_docs)} 个类别")
        for coll_name, docs in classified_docs.items():
            print(f"     - {coll_name}: {len(docs)} 个文档")

        # 为每个 collection 创建向量存储
        vector_stores = {}
        for collection_key, docs in classified_docs.items():
            # 获取实际的 collection name
            coll_config = self.config.get_collection_config(collection_key)
            collection_name = coll_config.name
            
            print(f"\n📦 处理集合 '{collection_name}' (key: {collection_key})...")
            print(f"   配置信息: name={coll_config.name}, description={coll_config.description}")
            
            print("✂️  分割文档...")
            chunks = self.split_documents(docs)
            print(f"   生成了 {len(chunks)} 个文本块")
            
            print("🔢 创建向量存储...")
            vector_store = self.create_vector_store(chunks, collection_name)
            if vector_store:
                vector_stores[collection_key] = vector_store  # 用 key 作为字典的键

        print("\n✅ 向量存储创建完成")

        # 恢复原始设置
        if use_smart_classification is not None:
            self.use_smart_classification = original_setting

        return vector_stores
