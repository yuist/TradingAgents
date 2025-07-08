# TradingAgents 主程序入口文件
# 该文件展示了如何使用 TradingAgents 框架进行交易决策

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.config_manager import get_config_manager
import copy
import os

if __name__ == "__main__":
    # 方式1: 使用新的配置文件系统（推荐）
    config_path = "config.yaml"  # 配置文件路径
    
    if os.path.exists(config_path):
        # 使用配置文件系统
        # 使用配置文件初始化
        trading_graph = TradingAgentsGraph(
            selected_analysts=["market", "social", "news", "fundamentals"],
            debug=True,
            config_path=config_path
        )
    else:
        # 配置文件不存在，使用传统配置方式
        # 方式2: 传统配置方式（向后兼容）
        config = copy.deepcopy(DEFAULT_CONFIG)
        
        # 自定义配置
        config["llm_provider"] = "google"  # 使用 Google 作为 LLM 提供商
        config["backend_url"] = ""  # Google 不需要自定义后端 URL
        config["deep_think_llm"] = "gemini-2.0-flash"  # 深度思考模型
        config["quick_think_llm"] = "gemini-2.0-flash"  # 快速思考模型
        config["max_debate_rounds"] = 3  # 最大辩论轮数
        config["use_online_tools"] = True  # 启用在线工具
        
        # 初始化交易智能体图
        trading_graph = TradingAgentsGraph(
            selected_analysts=["market", "social", "news", "fundamentals"],
            debug=True,
            config=config
        )
    
    # 执行交易决策
    final_state, decision = trading_graph.propagate("AAPL", "2024-01-15")
    # 交易决策结果已生成
