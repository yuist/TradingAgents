import os

# TradingAgents 框架的默认配置文件
# 该文件定义了整个交易系统的核心配置参数

DEFAULT_CONFIG = {
    # 项目根目录路径
    # 用于定位项目文件和资源
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    
    # 结果存储目录
    # 可通过环境变量 TRADINGAGENTS_RESULTS_DIR 覆盖，默认为 ./results
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    
    # 数据存储目录
    # 存储交易历史数据、市场数据等
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    
    # 数据缓存目录
    # 用于缓存频繁访问的数据，提高系统性能
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    
    # LLM（大语言模型）设置
    # 指定使用的 LLM 提供商，支持 openai、google 等
    "llm_provider": "openai",
    
    # 深度思考模型
    # 用于复杂分析和决策的 LLM 模型，需要更强的推理能力
    "deep_think_llm": "o4-mini",
    
    # 快速思考模型  
    # 用于简单任务和快速响应的 LLM 模型，注重响应速度
    "quick_think_llm": "gpt-4o-mini",
    
    # LLM API 后端地址
    # 可根据不同的提供商配置相应的 API 端点
    "backend_url": "https://api.openai.com/v1",
    
    # 辩论和讨论设置
    # 研究员之间最大辩论轮数，用于平衡分析深度和效率
    "max_debate_rounds": 1,
    
    # 风险管理团队最大讨论轮数
    # 控制风险评估的深度和时间
    "max_risk_discuss_rounds": 1,
    
    # 最大递归限制
    # 防止系统陷入无限循环
    "max_recur_limit": 100,
    
    # 工具设置
    # 是否使用在线工具获取实时数据
    # True: 使用实时数据接口
    # False: 使用缓存的历史数据
    "online_tools": True,
}
