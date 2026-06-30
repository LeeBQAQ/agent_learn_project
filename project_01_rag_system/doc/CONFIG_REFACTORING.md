# 配置重构说明

## 📋 概述

已将智能路由和智能分类的配置参数统一整合到 `RAGConfig` 中，实现配置集中化管理。

## ✅ 改进内容

### 1. RAGConfig 新增配置项

在 `module_02_config/config_01_rag_config.py` 中添加：

```python
# 智能功能配置
use_smart_routing: bool = True          # 是否启用智能查询路由
use_smart_classification: bool = True   # 是否启用智能文档分类
```

### 2. 简化的 API

**之前：**
```python
# 需要在多处传递参数
processor = DocumentProcessor(config, use_smart_classification=True)
rag = RAGChain(config, use_smart_routing=True)
```

**现在：**
```python
# 只需在配置中设置一次
config = RAGConfig(
    chunk_size=300,
    top_k=3,
    use_smart_routing=True,
    use_smart_classification=True
)

processor = DocumentProcessor(config)
rag = RAGChain(config)
```

## 🎯 优势

### 1. 配置集中化
- 所有配置都在一个地方管理
- 避免参数在不同层级传递
- 更容易理解和维护

### 2. 代码简化
- 减少了函数参数数量
- 调用更简洁
- 降低了出错概率

### 3. 灵活性
- 可以通过配置文件统一管理
- 支持运行时动态修改配置
- 便于测试不同配置组合

## 📝 使用示例

### 启用所有智能功能（推荐）

```python
config = RAGConfig(
    chunk_size=300,
    chunk_overlap=50,
    top_k=3,
    use_smart_routing=True,           # ✅ 启用
    use_smart_classification=True     # ✅ 启用
)
```

### 禁用智能功能（用于调试或性能优化）

```python
config = RAGConfig(
    chunk_size=300,
    chunk_overlap=50,
    top_k=3,
    use_smart_routing=False,          # ❌ 禁用
    use_smart_classification=False    # ❌ 禁用
)
```

### 混合使用

```python
config = RAGConfig(
    chunk_size=300,
    use_smart_routing=True,           # ✅ 只启用查询路由
    use_smart_classification=False    # ❌ 禁用文档分类
)
```

## 🔧 影响范围

以下文件已更新以使用新的配置方式：

1. ✅ `module_02_config/config_01_rag_config.py` - 添加配置项
2. ✅ `ragchain.py` - 从 config 读取配置
3. ✅ `module_03_document/document_loader.py` - 从 config 读取配置
4. ✅ `main.py` - 简化调用方式

## 💡 最佳实践

### 1. 生产环境
```python
config = RAGConfig(
    use_smart_routing=True,
    use_smart_classification=True
)
```

### 2. 开发/测试环境（快速迭代）
```python
config = RAGConfig(
    use_smart_routing=False,
    use_smart_classification=False
)
```

### 3. 性能敏感场景
```python
config = RAGConfig(
    use_smart_routing=True,      # 保留查询路由提高准确性
    use_smart_classification=False  # 禁用文档分类加快索引速度
)
```

## 🚀 迁移指南

如果您之前的代码使用了旧的 API：

**旧代码：**
```python
processor = DocumentProcessor(config, use_smart_classification=True)
rag = RAGChain(config, use_smart_routing=True)
```

**新代码：**
```python
# 在创建 config 时设置
config = RAGConfig(
    use_smart_routing=True,
    use_smart_classification=True
)

# 调用时不需要额外参数
processor = DocumentProcessor(config)
rag = RAGChain(config)
```

## ❓ 常见问题

**Q: 为什么要把参数放到 Config 中？**
A: 配置集中化是软件工程的最佳实践，便于管理和维护。

**Q: 还能临时覆盖配置吗？**
A: 可以，直接修改 config 对象的属性即可：
```python
config.use_smart_routing = False
```

**Q: 会影响性能吗？**
A: 不会，只是改变了参数的传递方式，功能完全相同。

---

**重构完成！** 🎉
