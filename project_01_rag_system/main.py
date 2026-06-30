from ragchain import RAGChain
from module_02_config.config_01_rag_config import RAGConfig, SmartMode
from module_03_document.document_loader import DocumentProcessor
from module_03_document.milvus_store import get_milvus_client

def read_file_content(file_path: str) -> str:
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"⚠️  读取文件 {file_path} 失败: {e}")
        return ""

def clear_milvus_collections(config: RAGConfig):
    """清除 Milvus 中的所有 collection"""
    try:
        print("🗑️  清理 Milvus 集合...")
        client = get_milvus_client(config)
        
        # 获取所有 collection
        collections = client.list_collections()
        
        if not collections:
            print("   ℹ️  没有需要清理的集合")
            return
        
        print(f"   发现 {len(collections)} 个集合: {collections}")
        
        # 删除所有 collection
        for coll_name in collections:
            try:
                client.drop_collection(coll_name)
                print(f"   ✅ 已删除集合: {coll_name}")
            except Exception as e:
                print(f"   ⚠️  删除集合 {coll_name} 失败: {e}")
        
        print("   ✅ 清理完成\n")
        
    except Exception as e:
        print(f"   ❌ 清理失败: {e}")
        print("   请确保 Milvus 服务正在运行\n")

def main():
    """主程序：演示 RAG 系统的使用"""
    print("=" * 60)
    print("🚀 RAG 检索增强生成系统演示")
    print("=" * 60)
    
    # 初始化config（启用智能功能）
    config = RAGConfig(
        chunk_size=300,
        chunk_overlap=50,
        top_k=3,
        smart_mode=SmartMode.FULL  # 启用所有智能功能
    )
    
    # 清除旧的 collection
    clear_milvus_collections(config)
    # 准备文档路径（按主题分类）
    file_paths = [
        # 编程类文档
        "temp/python_intro.txt",
        "temp/django_framework.md",
        "temp/flask_guide.md",
        # AI/ML 类文档
        "temp/ml_concepts.csv",
        "temp/neural_networks.csv",
        "temp/deep_learning_applications.csv",
        # 框架类文档
        "temp/langchain_guide.md",
    ]
    
    # 读取文件内容
    print("📦 读取文档...")
    documents = []
    metadatas = []
    for file_path in file_paths:
        content = read_file_content(file_path)
        if content:
            documents.append(content)
            metadatas.append({"source": file_path})
    
    if not documents:
        print("❌ 没有成功读取任何文档")
        return
    
    print(f"   成功读取 {len(documents)} 个文档\n")
    
    # 创建文档处理器并索引
    print("📦 索引文档...")
    processor = DocumentProcessor(config)
    vector_stores = processor.process(
        texts=documents,
        metadatas=metadatas,
        milvus_collection="demo"
    )
    
    print(f"\n✅ 文档已分类存储到 {len(vector_stores)} 个集合:")
    for coll_name in vector_stores.keys():
        print(f"   - {coll_name}")

    print("\n📦 初始化 RAG 系统...")
    rag = RAGChain(config)

    print("\n💡 提示：输入 'quit' 或 'exit' 退出程序\n")
    
    while True:
        try:
            query = input("请输入查询：").strip()
            
            # 退出条件
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 再见！")
                break
            
            # 空输入检查
            if not query:
                print("⚠️  请输入有效的查询内容\n")
                continue
            
            print("\n" + "=" * 60)
            print(f"🔍 查询：{query}")
            print("=" * 60)
            
            # 执行查询
            result = rag.query(query)
            
            # 显示结果
            print("\n✅ 回答：")
            print(result.get('answer', '无回答'))
            
            # 显示来源
            sources = result.get('sources', [])
            if sources:
                print("\n📚 参考来源：")
                for source in sources:
                    print(f"  - {source.get('source', 'unknown')}")
            
            # 显示置信度
            confidence = result.get('confidence', 0)
            print(f"\n🎯 置信度：{confidence:.2%}")
            print("\n" + "-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 程序已中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误：{str(e)}")
            print("请重试\n")

if __name__ == '__main__':
    main()