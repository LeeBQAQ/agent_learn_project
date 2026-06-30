"""
智能路由功能测试脚本
测试 LLM 是否能正确地将文档分类到不同的 collection，
以及是否能根据用户问题选择正确的 collection 进行检索
"""

from module_02_config.config_01_rag_config import RAGConfig
from module_03_document.document_loader import DocumentProcessor
from module_04_retriever.smart_router import SmartRouter, DocumentClassifier
import sys


def test_document_classification():
    """测试文档分类功能"""
    print("=" * 80)
    print("📄 测试 1: 文档分类功能")
    print("=" * 80)
    
    config = RAGConfig()
    classifier = DocumentClassifier(config)
    
    # 测试不同主题的文档
    test_documents = [
        ("Python 是一种广泛使用的编程语言...", "应该分类到 programming"),
        ("神经网络是深度学习的基础...", "应该分类到 ai_ml"),
        ("Django 是一个 Python Web 框架...", "应该分类到 frameworks 或 programming"),
        ("机器学习算法包括监督学习和无监督学习...", "应该分类到 ai_ml"),
        ("Flask 是一个轻量级的 Web 框架...", "应该分类到 frameworks"),
    ]
    
    for doc_content, expected in test_documents:
        print(f"\n📝 文档预览: {doc_content[:50]}...")
        result = classifier.classify_document(doc_content)
        print(f"   ✅ 分类结果: {result['collection']}")
        print(f"   🎯 置信度: {result['confidence']:.2f}")
        print(f"   💡 理由: {result['reason']}")
        print(f"   📌 预期: {expected}")
        print("-" * 80)


def test_query_routing():
    """测试查询路由功能"""
    print("\n" + "=" * 80)
    print("🔍 测试 2: 查询路由功能")
    print("=" * 80)
    
    config = RAGConfig()
    router = SmartRouter(config)
    
    # 测试不同类型的查询
    test_queries = [
        ("Python 有哪些特点？", "应该路由到 programming"),
        ("什么是神经网络？", "应该路由到 ai_ml"),
        ("如何使用 Django 创建项目？", "应该路由到 frameworks 或 programming"),
        ("机器学习的基本概念是什么？", "应该路由到 ai_ml"),
        ("Flask 和 Django 有什么区别？", "应该路由到 frameworks"),
        ("深度学习有哪些应用？", "应该路由到 ai_ml"),
    ]
    
    for query, expected in test_queries:
        print(f"\n❓ 查询: {query}")
        collections = router.route_query(query)
        print(f"   ✅ 选中集合: {collections}")
        print(f"   📌 预期: {expected}")
        print("-" * 80)


def test_collection_config():
    """测试集合配置"""
    print("\n" + "=" * 80)
    print("⚙️  测试 3: 集合配置")
    print("=" * 80)
    
    config = RAGConfig()
    
    print("\n📋 可用的集合:")
    for key, coll_config in config.collections.items():
        print(f"   - {key}:")
        print(f"     名称: {coll_config.name}")
        print(f"     描述: {coll_config.description}")
        print(f"     Top-K: {coll_config.top_k}")
        print()


def main():
    """运行所有测试"""
    print("\n🚀 开始智能路由功能测试\n")
    
    try:
        # 测试集合配置
        test_collection_config()
        
        # 测试文档分类
        test_document_classification()
        
        # 测试查询路由
        test_query_routing()
        
        print("\n" + "=" * 80)
        print("✅ 所有测试完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
