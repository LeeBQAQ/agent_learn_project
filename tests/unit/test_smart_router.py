from unittest.mock import MagicMock

from src.core.config import RAGConfig
from src.core.smart_router import DocumentClassifier


def test_classifier_fallback_on_error():
    config = RAGConfig()
    classifier = DocumentClassifier(config)
    classifier.classifier_chain = MagicMock()
    classifier.classifier_chain.invoke.side_effect = RuntimeError("LLM error")
    result = classifier.classify_document("Python 是编程语言")
    assert result["collection"] == "default"
    assert result["confidence"] == 0.0


def test_classifier_parses_json():
    config = RAGConfig()
    classifier = DocumentClassifier(config)
    classifier.classifier_chain = MagicMock()
    classifier.classifier_chain.invoke.return_value = (
        '{"collection": "programming", "confidence": 0.9, "reason": "编程内容"}'
    )
    result = classifier.classify_document("Python list comprehension")
    assert result["collection"] == "programming"
    assert result["confidence"] == 0.9
