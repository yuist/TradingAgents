# TradingAgents 主程序入口文件
# 该文件展示了如何使用 TradingAgents 框架进行交易决策

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# 创建自定义配置
# 通过复制默认配置并修改特定参数来自定义系统行为
config = DEFAULT_CONFIG.copy()

# 配置 LLM 提供商为 Google
# 可选值：openai、google、anthropic 等
config["llm_provider"] = "google"  # Use a different model

# 配置后端 API 地址
# 根据不同的 LLM 提供商设置相应的 API 端点
config["backend_url"] = "https://generativelanguage.googleapis.com/v1"  # Use a different backend

# 配置深度思考模型
# 用于需要深入分析的任务，如基本面分析、风险评估等
config["deep_think_llm"] = "gemini-2.0-flash"  # Use a different model

# 配置快速思考模型
# 用于需要快速响应的任务，如实时数据处理、简单计算等
config["quick_think_llm"] = "gemini-2.0-flash"  # Use a different model

# 设置最大辩论轮数
# 控制研究员智能体之间的辩论深度
# 增加轮数可以获得更全面的分析，但会增加时间成本
config["max_debate_rounds"] = 1  # Increase debate rounds

# 启用在线工具
# True: 使用实时市场数据和新闻
# False: 使用缓存的历史数据（用于回测）
config["online_tools"] = True  # Increase debate rounds

# 初始化交易智能体图
# debug=True 会输出详细的执行过程信息
ta = TradingAgentsGraph(debug=True, config=config)

# 执行交易决策流程
# 参数1: 股票代码（如 NVDA 代表英伟达）
# 参数2: 交易日期（格式：YYYY-MM-DD）
# 返回值: (内部状态, 交易决策)
_, decision = ta.propagate("NVDA", "2024-05-10")

# 输出最终的交易决策
print(decision)

# 反思和记忆功能（当前已注释）
# 该功能允许系统从过去的交易中学习
# 参数：仓位收益（正数表示盈利，负数表示亏损）
# ta.reflect_and_remember(1000) # parameter is the position returns
