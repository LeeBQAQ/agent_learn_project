from src.core.config import RAGConfig, SmartMode
from src.core.document_loader import DocumentProcessor


def test_load_documents(test_config):
    processor = DocumentProcessor(test_config)
    texts = ["Hello world", "Python programming"]
    docs = processor.load_documents(texts)
    assert len(docs) == 2
    assert docs[0].page_content == "Hello world"
    assert docs[0].metadata["source"] == "doc_0"


def test_load_documents_with_metadata(test_config):
    processor = DocumentProcessor(test_config)
    texts = ["Content A"]
    metas = [{"source": "file.txt"}]
    docs = processor.load_documents(texts, metas)
    assert docs[0].metadata["source"] == "file.txt"


def test_split_documents(test_config):
    processor = DocumentProcessor(test_config)
    docs = processor.load_documents(["第一段内容。\n\n第二段内容。"])
    chunks = processor.split_documents(docs)
    assert len(chunks) >= 1


def test_classify_without_smart_mode():
    config = RAGConfig(smart_mode=SmartMode.DISABLED)
    processor = DocumentProcessor(config)
    texts = ["Python 编程", "机器学习算法"]
    docs = processor.load_documents(texts)
    classified = processor.classify_and_store(docs, "default")
    assert "default" in classified
    assert len(classified["default"]) == 2
