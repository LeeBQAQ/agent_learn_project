# 智能路由功能使用指南

## 📋 概述

本 RAG 系统现已支持**智能路由功能**，能够自动：
1. **文档分类**：根据文档内容自动将其存储到合适的 collection
2. **查询路由**：根据用户问题自动选择最相关的 collection 进行检索

## 🗂️ 集合配置

系统预定义了以下集合（collections）：

### 1. default（默认集合）
- **名称**: `rag_documents`
- **描述**: 存放通用知识
- **用途**: 未明确分类的文档

### 2. programming（编程类）
- **名称**: `programming_docs`
- **描述**: 编程语言和技术文档
- **包含**: Python、Java 等编程语言相关知识
- **示例文档**: 
  - `python_intro.txt` - Python 介绍
  - `django_framework.md` - Django 框架
  - `flask_guide.md` - Flask 框架

### 3. ai_ml（人工智能/机器学习）
- **名称**: `ai_ml_docs`
- **描述**: AI 和 ML 相关文档
- **包含**: 算法、框架、概念等
- **示例文档**:
  - `ml_concepts.csv` - 机器学习概念
  - `neural_networks.csv` - 神经网络基础
  - `deep_learning_applications.csv` - 深度学习应用

### 4. frameworks（开发框架）
- **名称**: `framework_docs`
- **描述**: 开发框架和工具
- **包含**: LangChain、Django、Flask 等
- **示例文档**:
  - `langchain_guide.md` - LangChain 指南

## 🚀 快速开始

### 1. 运行测试脚本

在正式使用前，建议先运行测试脚本验证智能路由功能：

```bash
python test_smart_routing.py
```

这将测试：
- ✅ 集合配置是否正确
- ✅ 文档分类是否准确
- ✅ 查询路由是否智能

### 2. 运行主程序

```bash
python main.py
```

程序会：
1. 读取 temp 文件夹中的所有文档
2. 使用 LLM 自动分类文档并存储到对应 collection
3. 启动交互式查询界面
4. 根据用户问题智能选择 collection 进行检索

### 3. 测试查询示例

尝试以下类型的查询来测试智能路由：

**编程类查询：**
```
- Python 有哪些特点？
- 如何定义 Python 函数？
- Python 支持哪些数据类型？
```

**AI/ML 类查询：**
```
- 什么是神经网络？
- 机器学习有哪些算法？
- 深度学习的应用场景有哪些？
```

**框架类查询：**
```
- LangChain 是什么？
- Django 和 Flask 有什么区别？
- 如何使用 Flask 创建 API？
```

## 🔧 配置说明

### 添加新的集合

在 `module_02_config/config_01_rag_config.py` 中添加：

```python
collections: Dict[str, CollectionConfig] = field(default_factory=lambda: {
    # ... 现有集合 ...
    
    "your_category": CollectionConfig(
        name="your_collection_name",
        description="集合描述",
        top_k=3
    ),
})
```

### 启用/禁用智能路由

**文档处理时：**
```python
# 启用智能分类
processor = DocumentProcessor(config, use_smart_classification=True)

# 禁用智能分类（使用默认集合）
processor = DocumentProcessor(config, use_smart_classification=False)
```

**查询时：**
```python
# 启用智能路由
rag = RAGChain(config, use_smart_routing=True)

# 禁用智能路由（使用默认集合）
rag = RAGChain(config, use_smart_routing=False)
```

## 📊 工作流程

### 文档存储流程

```
📄 文档加载
    ↓
✂️  文本分块
    ↓
🤖 LLM 分析内容
    ↓
🎯 智能分类
    ↓
📦 存储到对应 collection
```

### 查询检索流程

```
❓ 用户提问
    ↓
🤖 LLM 分析问题
    ↓
🎯 智能路由选择 collection
    ↓
🔍 从选中的 collection 检索
    ↓
🔄 合并多个 collection 的结果
    ↓
✨ 去重并排序
    ↓
💬 生成回答
```

## 💡 最佳实践

### 1. 文档组织
- 将相关主题的文档放在 temp 文件夹
- 文件名应反映文档主题
- 保持文档内容清晰、结构化

### 2. 查询优化
- 问题应该具体明确
- 避免过于模糊的问题
- 可以追问以获取更详细的信息

### 3. 性能调优
- 调整 `top_k` 参数控制检索数量
- 根据文档类型调整 `chunk_size` 和 `chunk_overlap`
- 监控智能路由的准确性，必要时调整提示词

## 🔍 故障排查

### 问题 1: 文档分类不准确

**可能原因：**
- 文档内容太短或不够清晰
- 提示词需要优化

**解决方案：**
- 增加文档内容的详细描述
- 修改 `DOCUMENT_CLASSIFIER_PROMPT` 提示词

### 问题 2: 查询路由错误

**可能原因：**
- 问题表述不清
- 集合描述不够明确

**解决方案：**
- 重新表述问题，使其更具体
- 完善 `CollectionConfig` 中的 description

### 问题 3: 检索结果为空

**可能原因：**
- 对应 collection 中没有相关文档
- 路由选择了错误的 collection

**解决方案：**
- 检查文档是否正确分类和存储
- 查看路由日志，确认选择的 collection
- 考虑使用多 collection 检索

## 📝 示例输出

### 文档分类示例

```
📄 文档分类: programming (置信度: 0.92)
   理由: 文档主要介绍 Python 编程语言的特性和应用

📄 文档分类: ai_ml (置信度: 0.88)
   理由: 文档包含机器学习和神经网络的相关概念
```

### 查询路由示例

```
❓ 查询: Python 有哪些特点？
🎯 智能路由选中集合: ['programming']
📚 检索到 5 个相关文档

❓ 查询: 什么是神经网络？
🎯 智能路由选中集合: ['ai_ml']
📚 检索到 7 个相关文档

❓ 查询: Django 和 Flask 有什么区别？
🎯 智能路由选中集合: ['frameworks', 'programming']
📚 检索到 10 个相关文档
```

## 🎓 学习资源

- [LangChain 官方文档](https://python.langchain.com/)
- [Milvus 向量数据库](https://milvus.io/)
- [智能路由设计原则](README.md)

## ❓ 常见问题

**Q: 智能路由会影响性能吗？**
A: 会有少量额外开销（每次调用 LLM），但能显著提高检索准确性。对于生产环境，可以考虑缓存路由结果。

**Q: 可以自定义集合数量吗？**
A: 可以！在配置文件中添加或删除集合即可。

**Q: 如果路由失败怎么办？**
A: 系统会自动降级到默认集合，确保始终有结果返回。

**Q: 如何查看路由决策过程？**
A: 运行时会在控制台打印路由日志，包括选中的集合和置信度。

---

**祝您使用愉快！** 🎉
