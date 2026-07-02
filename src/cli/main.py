from src.core.config import RAGConfig, SmartMode
from src.core.document_loader import DocumentProcessor
from src.core.milvus_store import get_milvus_client
from src.core.rag_chain import RAGChain


def read_file_content(file_path: str) -> str:
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: 读取文件 {file_path} 失败: {e}")
        return ""


def clear_milvus_collections(config: RAGConfig):
    try:
        client = get_milvus_client(config)
        collections = client.list_collections()
        if not collections:
            return
        for coll_name in collections:
            try:
                client.drop_collection(coll_name)
            except Exception:
                pass
    except Exception:
        pass


def main():
    config = RAGConfig(chunk_size=300, chunk_overlap=50, top_k=3, smart_mode=SmartMode.FULL)
    clear_milvus_collections(config)

    file_paths = [
        "legacy/project_01_rag_system/temp/python_intro.txt",
        "legacy/project_01_rag_system/temp/django_framework.md",
        "legacy/project_01_rag_system/temp/flask_guide.md",
        "legacy/project_01_rag_system/temp/ml_concepts.csv",
        "legacy/project_01_rag_system/temp/neural_networks.csv",
        "legacy/project_01_rag_system/temp/deep_learning_applications.csv",
        "legacy/project_01_rag_system/temp/langchain_guide.md",
    ]

    documents = []
    metadatas = []
    for fp in file_paths:
        content = read_file_content(fp)
        if content:
            documents.append(content)
            metadatas.append({"source": fp})

    if not documents:
        print("没有成功读取任何文档")
        return

    processor = DocumentProcessor(config)
    vector_stores = processor.process(texts=documents, metadatas=metadatas, milvus_collection="demo")
    print(f"文档已分类存储到 {len(vector_stores)} 个集合")

    rag = RAGChain(config)
    print("输入 'quit' 退出")

    while True:
        try:
            query = input("请输入查询：").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            result = rag.query(query)
            print(f"回答：{result.get('answer', '无回答')}")
            sources = result.get("sources", [])
            if sources:
                print("参考来源：")
                for s in sources:
                    print(f"  - {s.get('source', 'unknown')}")
            print(f"置信度：{result.get('confidence', 0):.2%}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误：{e}")


if __name__ == "__main__":
    main()
