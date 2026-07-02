import json
from typing import Any

from langchain_core.output_parsers import StrOutputParser

from src.core.config import RAGConfig
from src.core.model import model
from src.core.templates import TemplateLibrary


class SmartRouter:
    """智能路由器：使用 LLM 决定 collection 选择"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.router_chain = TemplateLibrary.COLLECTION_ROUTER_PROMPT | model | StrOutputParser()

    def get_collections_info(self) -> str:
        info_lines = []
        for key, coll_config in self.config.collections.items():
            info_lines.append(f"- {key}: {coll_config.description or coll_config.name}")
        return "\n".join(info_lines)

    def route_query(self, query: str) -> list[str]:
        """根据查询内容，智能选择 collection"""
        try:
            result = self.router_chain.invoke({"query": query, "collections_info": self.get_collections_info()})
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            selected = json.loads(result.strip())
            if not isinstance(selected, list):
                selected = [selected]
            valid = [c for c in selected if c in self.config.collections]
            if not valid:
                return ["default"]
            return valid
        except Exception:
            return ["default"]


class DocumentClassifier:
    """文档分类器：使用 LLM 自动分类文档"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.classifier_chain = TemplateLibrary.DOCUMENT_CLASSIFIER_PROMPT | model | StrOutputParser()

    def get_categories_info(self) -> str:
        info_lines = []
        for key, coll_config in self.config.collections.items():
            info_lines.append(f"- {key}: {coll_config.description or coll_config.name}")
        return "\n".join(info_lines)

    def classify_document(self, document_content: str) -> dict[str, Any]:
        """分类文档"""
        try:
            result = self.classifier_chain.invoke(
                {"document_content": document_content[:2000], "collections_info": self.get_categories_info()}
            )
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            classification = json.loads(result.strip())
            return {
                "collection": classification.get("collection", "default"),
                "confidence": classification.get("confidence", 0.5),
                "reason": classification.get("reason", ""),
            }
        except Exception as e:
            return {"collection": "default", "confidence": 0.0, "reason": str(e)}
