from typing import List, Dict, Any
from module_05_generator.template_library import TemplateLibrary
from module_02_config.config_01_rag_config import RAGConfig
from model.model_init import model
from langchain_core.output_parsers import StrOutputParser
import json


class SmartRouter:
    """智能路由器：使用 LLM 决定 collection 选择"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = model
        self.router_chain = TemplateLibrary.COLLECTION_ROUTER_PROMPT | model | StrOutputParser()

    def get_collections_info(self) -> str:
        """获取所有 collection 的信息"""
        collections = self.config.collections
        info_lines = []
        for key, coll_config in collections.items():
            info_lines.append(f"- {key}: {coll_config.description or coll_config.name}")
        return "\n".join(info_lines)

    def route_query(self, query: str) -> List[str]:
        """根据查询内容，智能选择 collection
        
        Args:
            query: 用户查询
            
        Returns:
            选中的 collection 名称列表
        """
        try:
            collections_info = self.get_collections_info()
            
            result = self.router_chain.invoke({
                "query": query,
                "collections_info": collections_info
            })
            
            # 解析 JSON 结果
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            selected_collections = json.loads(result)
            
            # 验证结果
            if not isinstance(selected_collections, list):
                selected_collections = [selected_collections]
            
            # 过滤不存在的 collection
            valid_collections = [c for c in selected_collections if c in self.config.collections]
            
            if not valid_collections:
                print(f"⚠️  路由未选中有效集合，使用默认集合")
                return ["default"]
            
            print(f"🎯 智能路由选中集合: {valid_collections}")
            return valid_collections
            
        except Exception as e:
            print(f"⚠️  路由失败: {e}，使用默认集合")
            return ["default"]


class DocumentClassifier:
    """文档分类器：使用 LLM 自动分类文档"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = model
        self.classifier_chain = TemplateLibrary.DOCUMENT_CLASSIFIER_PROMPT | model | StrOutputParser()

    def get_categories_info(self) -> str:
        """获取分类信息"""
        collections = self.config.collections
        info_lines = []
        for key, coll_config in collections.items():
            info_lines.append(f"- {key}: {coll_config.description or coll_config.name}")
        return "\n".join(info_lines)

    def classify_document(self, document_content: str) -> Dict[str, Any]:
        """分类文档
        
        Args:
            document_content: 文档内容
            
        Returns:
            分类结果字典
        """
        try:
            collections_info = self.get_categories_info()
            
            result = self.classifier_chain.invoke({
                "document_content": document_content[:2000],  # 限制长度
                "collections_info": collections_info
            })
            
            # 解析 JSON 结果
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            classification = json.loads(result)
            
            return {
                "collection": classification.get("collection", "default"),
                "confidence": classification.get("confidence", 0.5),
                "reason": classification.get("reason", "")
            }
            
        except Exception as e:
            print(f"⚠️  文档分类失败: {e}")
            return {
                "collection": "default",
                "confidence": 0.0,
                "reason": f"分类错误: {str(e)}"
            }
