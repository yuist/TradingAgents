# Qwen模型配置指南

本指南详细介绍了如何在TradingAgents项目中优化配置和使用Qwen模型的各种功能。

## 概述

Qwen模型支持多种高级功能，包括：
- 联网搜索 (`enable_search`)
- 思考模式 (`enable_thinking`)
- 搜索策略优化 (`search_options`)

## 配置管理器

### QwenConfigManager类

`QwenConfigManager` 提供了统一的Qwen模型配置管理：

```python
from tradingagents.dataflows.qwen_config import QwenConfigManager

# 获取新闻搜索的优化配置
config = QwenConfigManager.get_news_search_config(
    model="qwen-plus",
    task_type="stock_news"  # 或 "global_news"
)
```

### 支持的模型

#### 联网搜索支持
- `qwen-plus`
- `qwen-turbo` 
- `qwen-max`
- `qwen3`

#### 思考模式支持
- `qwen3`
- `qwen-plus`
- `qwen-turbo`
- `qwen-max`

## 配置参数详解

### enable_search (联网搜索)

```python
extra_body = {
    "enable_search": True  # 启用联网搜索
}
```

**作用**: 模型在生成文本时使用互联网搜索结果进行参考

**注意事项**:
- 启用后可能增加Token消耗
- 如果模型没有搜索，建议设置 `forced_search: True`

### search_options (搜索策略)

```python
extra_body = {
    "enable_search": True,
    "search_options": {
        "forced_search": True,      # 强制开启搜索
        "search_strategy": "pro"    # "standard" 或 "pro"
    }
}
```

**搜索策略对比**:
- `"standard"`: 搜索5条互联网信息
- `"pro"`: 搜索10条互联网信息（更全面）

### enable_thinking (思考模式)

```python
extra_body = {
    "enable_thinking": False  # 商业版建议关闭以提高速度
}
```

**默认值**:
- Qwen3商业版: `False`
- Qwen3开源版: `True`

## 任务类型优化

### 股票新闻搜索

```python
# 针对股票新闻优化的配置
config = QwenConfigManager.get_news_search_config(
    model="qwen-plus",
    task_type="stock_news"
)
# 结果: {"enable_search": True, "search_options": {"forced_search": True, "search_strategy": "standard"}, "enable_thinking": False}
```

**特点**:
- 使用标准搜索策略（5条信息）
- 关闭思考模式以提高响应速度
- 强制搜索确保获取最新信息

### 全球新闻搜索

```python
# 针对全球新闻优化的配置
config = QwenConfigManager.get_news_search_config(
    model="qwen-plus",
    task_type="global_news"
)
# 结果: {"enable_search": True, "search_options": {"forced_search": True, "search_strategy": "pro"}, "enable_thinking": False}
```

**特点**:
- 使用pro搜索策略（10条信息）
- 获取更全面的全球市场信息
- 适合复杂的宏观经济分析

## 实际使用示例

### 在interface.py中的应用

```python
from .qwen_config import QwenConfigManager

def get_stock_news_openai(symbol: str, curr_date: str) -> str:
    # ... 其他代码 ...
    
    if provider == "qwen":
        # 使用配置管理器获取优化配置
        extra_body_config = QwenConfigManager.get_news_search_config(
            model=model,
            task_type="stock_news"
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[...],
            extra_body=extra_body_config
        )
```

### 自定义配置

```python
# 获取自定义配置
custom_config = QwenConfigManager.get_extra_body_config(
    model="qwen-plus",
    enable_search=True,
    enable_thinking=False,
    forced_search=True,
    search_strategy="pro"
)
```

## 模型能力检查

```python
# 检查模型支持的功能
capabilities = QwenConfigManager.get_model_capabilities("qwen-plus")
print(capabilities)
# 输出: {'supports_search': True, 'supports_thinking': True, 'is_open_source': False}
```

## 最佳实践

### 1. 任务类型选择
- **股票新闻**: 使用 `"stock_news"` 配置，平衡速度和准确性
- **全球新闻**: 使用 `"global_news"` 配置，获取更全面信息

### 2. 模型选择
- **高频交易**: 推荐 `qwen-turbo` + 标准搜索
- **深度分析**: 推荐 `qwen-plus` + pro搜索
- **复杂研究**: 推荐 `qwen-max` + pro搜索 + 思考模式

### 3. 性能优化
- 新闻搜索任务建议关闭思考模式
- 根据信息需求选择合适的搜索策略
- 使用强制搜索确保获取最新信息

### 4. 错误处理
```python
try:
    config = QwenConfigManager.get_news_search_config(model, task_type)
    # 使用配置进行API调用
except Exception as e:
    # 降级到基础配置
    config = {"enable_search": True}
```

## 配置验证

运行测试脚本验证配置：

```bash
python test_qwen_enhanced.py
```

测试将验证：
- 配置管理器功能
- 不同模型的能力检查
- 实际API调用效果
- 搜索质量指标

## 注意事项

1. **Token消耗**: 启用联网搜索会增加Token使用量
2. **响应时间**: pro搜索策略可能增加响应时间
3. **模型兼容性**: 确保使用支持相应功能的模型版本
4. **API限制**: 注意API调用频率限制

## 故障排除

### 搜索功能不工作
1. 检查模型是否支持联网搜索
2. 确认 `enable_search: True` 正确设置
3. 尝试启用 `forced_search: True`

### 响应速度慢
1. 关闭思考模式 `enable_thinking: False`
2. 使用标准搜索策略 `search_strategy: "standard"`
3. 考虑使用更快的模型如 `qwen-turbo`

### 搜索结果质量差
1. 启用pro搜索策略 `search_strategy: "pro"`
2. 优化prompt描述
3. 使用强制搜索 `forced_search: True`