# 智能功能模式说明

## 📋 概述

已将智能路由和智能分类的两个独立配置合并为一个 `smart_mode` 枚举，简化配置管理。

## 🎯 设计理念

### 之前的问题
```python
# 需要设置两个参数，容易混淆
use_smart_routing: bool = True
use_smart_classification: bool = True
```

**问题：**
- ❌ 配置分散，不够直观
- ❌ 可能出现不一致的状态（如只启用一个）
- ❌ 增加了配置的复杂度

### 现在的方案
```python
# 一个参数控制所有智能功能
smart_mode: SmartMode = SmartMode.FULL
```

**优势：**
- ✅ 配置集中，一目了然
- ✅ 预定义的模式，避免错误组合
- ✅ 更符合用户思维模式

## 🔧 三种模式

### 1. SmartMode.FULL（推荐）
**启用所有智能功能**
- ✅ 智能查询路由
- ✅ 智能文档分类

```python
config = RAGConfig(
    smart_mode=SmartMode.FULL
)
```

**适用场景：**
- 生产环境
- 追求最佳效果
- 不介意额外的 LLM 调用开销

---

### 2. SmartMode.ROUTING_ONLY
**仅启用查询路由**
- ✅ 智能查询路由
- ❌ 禁用文档分类（使用默认集合）

```python
config = RAGConfig(
    smart_mode=SmartMode.ROUTING_ONLY
)
```

**适用场景：**
- 文档已经手动分类好
- 只需要智能检索
- 想减少索引时的 LLM 调用

---

### 3. SmartMode.DISABLED
**禁用所有智能功能**
- ❌ 禁用查询路由（使用默认集合）
- ❌ 禁用文档分类（使用默认集合）

```python
config = RAGConfig(
    smart_mode=SmartMode.DISABLED
)
```

**适用场景：**
- 调试和测试
- 性能敏感场景
- 所有文档都在一个集合中

## 💡 使用示例

### 示例 1：生产环境（推荐）

```python
from module_02_config.config_01_rag_config import RAGConfig, SmartMode

config = RAGConfig(
    chunk_size=512,
    chunk_overlap=100,
    top_k=3,
    smart_mode=SmartMode.FULL  # 启用所有智能功能
)

processor = DocumentProcessor(config)
rag = RAGChain(config)
```

### 示例 2：快速原型开发

```python
config = RAGConfig(
    smart_mode=SmartMode.DISABLED  # 禁用智能功能，加快速度
)
```

### 示例 3：混合场景

```python
# 文档已预先分类，只需智能路由
config = RAGConfig(
    smart_mode=SmartMode.ROUTING_ONLY
)
```

## 🔍 内部实现

配置类内部通过属性自动转换：

```python
@property
def use_smart_routing(self) -> bool:
    """是否启用智能查询路由"""
    return self.smart_mode in [SmartMode.ROUTING_ONLY, SmartMode.FULL]

@property
def use_smart_classification(self) -> bool:
    """是否启用智能文档分类"""
    return self.smart_mode == SmartMode.FULL
```

这样其他代码不需要修改，仍然可以使用 `config.use_smart_routing` 和 `config.use_smart_classification`。

## 📊 对比表格

| 模式 | 查询路由 | 文档分类 | LLM 调用次数 | 适用场景 |
|------|---------|---------|-------------|---------|
| FULL | ✅ | ✅ | 多 | 生产环境 |
| ROUTING_ONLY | ✅ | ❌ | 中 | 已分类文档 |
| DISABLED | ❌ | ❌ | 少 | 调试/测试 |

## 🚀 迁移指南

### 旧代码
```python
config = RAGConfig(
    use_smart_routing=True,
    use_smart_classification=True
)
```

### 新代码
```python
config = RAGConfig(
    smart_mode=SmartMode.FULL
)
```

## ❓ 常见问题

**Q: 为什么不直接用布尔值？**
A: 枚举更清晰地表达了意图，避免了多个布尔值的组合混乱。

**Q: 能否自定义模式？**
A: 可以扩展 `SmartMode` 枚举添加新模式。

**Q: 会影响现有代码吗？**
A: 不会，内部属性保持了向后兼容。

**Q: 哪种模式性能最好？**
A: `DISABLED` 最快，但效果最差；`FULL` 最慢，但效果最好。

## 💭 设计思考

### 为什么选择枚举而不是单个布尔值？

1. **可扩展性** - 未来可以轻松添加新模式（如 `CLASSIFICATION_ONLY`）
2. **语义清晰** - `FULL` 比 `True + True` 更易理解
3. **类型安全** - 枚举限制了可选值，避免无效组合
4. **文档化** - 枚举本身就是文档，说明了所有可能的模式

### 为什么保留三个模式？

虽然目前主要使用 `FULL` 和 `DISABLED`，但 `ROUTING_ONLY` 为以下场景预留了空间：
- 用户已有分类好的文档库
- 只想优化检索，不想重新索引
- 分阶段启用智能功能

---

**配置更简洁，使用更灵活！** 🎉
