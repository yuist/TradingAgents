# Qwen 大模型支持文档

## 概述

TradingAgents 框架现已完全支持阿里云通义千问（Qwen）大模型系列。Qwen 是阿里巴巴达摩院开发的大规模语言模型，在中文理解和金融领域表现优异。

## 支持的模型

### 主要模型

| 模型名称 | 描述 | 适用场景 | 成本 |
|---------|------|----------|------|
| `qwen-turbo` | 快速响应模型 | 快速任务、实时分析 | 低 |
| `qwen-plus` | 平衡性能模型 | 一般分析任务 | 中 |
| `qwen-max` | 高性能模型 | 复杂分析、深度思考 | 高 |

### Qwen 2.5 系列

| 模型名称 | 参数规模 | 描述 |
|---------|----------|------|
| `qwen2.5-7b-instruct` | 7B | 轻量级指令模型 |
| `qwen2.5-14b-instruct` | 14B | 中等规模指令模型 |
| `qwen2.5-32b-instruct` | 32B | 大规模指令模型 |
| `qwen2.5-72b-instruct` | 72B | 超大规模指令模型 |

## 配置方法

### 1. 获取 API 密钥

1. 访问 [阿里云控制台](https://dashscope.console.aliyun.com/)
2. 开通 DashScope 服务
3. 创建 API Key

### 2. 配置文件设置

在 `config.yaml` 中添加以下配置：

```yaml
api_keys:
  qwen: 'your-qwen-api-key-here'  # 替换为您的实际 API 密钥

llm_providers:
  qwen:
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    models:
      deep:  # 用于复杂分析的模型
        - qwen-turbo
        - qwen-plus
        - qwen-max
        - qwen2.5-72b-instruct
        - qwen2.5-32b-instruct
      quick:  # 用于快速任务的模型
        - qwen-turbo
        - qwen2.5-7b-instruct
        - qwen2.5-14b-instruct

models:
  llm_provider: qwen
  deep_think_llm: qwen-max      # 深度思考模型
  quick_think_llm: qwen-turbo   # 快速思考模型
```

### 3. 环境变量设置（可选）

您也可以通过环境变量设置 API 密钥：

```bash
export QWEN_API_KEY="your-qwen-api-key-here"
```

## 使用示例

### 基本使用

```python
from tradingagents.config_manager import get_config_manager
from tradingagents.llm_factory import create_llm
from langchain.schema import HumanMessage

# 创建 Qwen 模型实例
config_manager = get_config_manager()
qwen_llm = create_llm(
    provider='qwen',
    model_name='qwen-turbo',
    config_manager=config_manager
)

# 发送消息
message = HumanMessage(content="分析一下当前的股市趋势")
response = qwen_llm.invoke([message])
print(response.content)
```

### 在 TradingAgents 中使用

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph

# 创建交易图实例，使用 Qwen 模型
trading_graph = TradingAgentsGraph(
    config_path="config.yaml",  # 包含 Qwen 配置的文件
    selected_analysts=["market", "news", "fundamentals"]
)

# 执行交易分析
result = trading_graph.run(
    stock_symbol="AAPL",
    query="分析苹果公司的投资价值"
)
```

## 性能建议

### 模型选择建议

1. **快速任务**：使用 `qwen-turbo` 或 `qwen2.5-7b-instruct`
   - 实时数据处理
   - 简单的市场分析
   - 快速响应场景

2. **平衡任务**：使用 `qwen-plus` 或 `qwen2.5-14b-instruct`
   - 一般的投资分析
   - 风险评估
   - 策略建议

3. **复杂任务**：使用 `qwen-max` 或 `qwen2.5-72b-instruct`
   - 深度基本面分析
   - 复杂的量化策略
   - 多维度综合分析

### 成本优化

1. **混合使用**：在配置中设置不同复杂度的任务使用不同模型
   ```yaml
   models:
     llm_provider: qwen
     deep_think_llm: qwen-max      # 复杂分析
     quick_think_llm: qwen-turbo   # 快速任务
   ```

2. **批量处理**：将多个相似任务合并处理以减少 API 调用次数

3. **缓存策略**：对于重复的分析任务，使用缓存机制

## 特性优势

### 中文支持
- Qwen 对中文的理解和生成能力优异
- 适合分析中文财经新闻和报告
- 支持中文金融术语和概念

### 金融领域优化
- 在金融数据分析方面表现出色
- 理解复杂的金融概念和关系
- 能够生成专业的投资建议

### 成本效益
- 相比国外模型，在中国地区访问速度更快
- 定价相对合理
- 支持多种规模的模型选择

## 故障排除

### 常见问题

1. **API 密钥错误**
   ```
   错误：未找到 qwen 的 API 密钥
   解决：检查 config.yaml 中的 api_keys.qwen 设置
   ```

2. **模型不可用**
   ```
   错误：模型验证失败
   解决：确认模型名称正确，检查 llm_providers.qwen.models 配置
   ```

3. **网络连接问题**
   ```
   错误：连接超时
   解决：检查网络连接，确认可以访问 dashscope.aliyuncs.com
   ```

### 调试方法

1. **运行测试脚本**
   ```bash
   python test_qwen_support.py
   ```

2. **查看日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **验证配置**
   ```python
   from tradingagents.config_manager import get_config_manager
   config = get_config_manager()
   print(config.list_available_models('qwen'))
   ```

## 更新日志

### v1.0.0 (当前版本)
- ✅ 添加 Qwen 基础支持
- ✅ 支持所有主要 Qwen 模型
- ✅ 完整的配置管理
- ✅ API 密钥管理
- ✅ 模型验证功能
- ✅ 错误处理和日志记录
- ✅ 使用示例和文档

## 相关链接

- [Qwen 官方文档](https://help.aliyun.com/zh/dashscope/)
- [DashScope API 文档](https://help.aliyun.com/zh/dashscope/developer-reference/)
- [TradingAgents 主文档](../README.md)
- [配置文件示例](../config_example_with_keys.yaml)

## 技术支持

如果您在使用 Qwen 模型时遇到问题，请：

1. 查看本文档的故障排除部分
2. 运行测试脚本进行诊断
3. 查看项目的 Issues 页面
4. 提交新的 Issue 并包含详细的错误信息

---

*最后更新：2024年12月*