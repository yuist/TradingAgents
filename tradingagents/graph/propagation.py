# TradingAgents/graph/propagation.py
# 传播模块：负责初始化智能体状态并在图中传播信息

from typing import Dict, Any
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """处理智能体图中的状态初始化和信息传播。"""

    def __init__(self, max_recur_limit=100):
        """使用配置参数初始化传播器。
        
        Args:
            max_recur_limit: 最大递归限制，防止无限循环
        """
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self, company_name: str, trade_date: str
    ) -> Dict[str, Any]:
        """为智能体图创建初始状态。
        
        Args:
            company_name: 目标公司名称或股票代码
            trade_date: 交易日期
            
        Returns:
            包含初始状态的字典，包括公司信息、交易日期和各种报告的空容器
        """
        return {
            "messages": [("human", company_name)],  # 初始消息
            "company_of_interest": company_name,   # 目标公司
            "trade_date": str(trade_date),         # 交易日期
            "investment_debate_state": InvestDebateState(
                {"history": "", "current_response": "", "count": 0}
            ),  # 投资辩论状态，用于跟踪看涨/看跌研究员的辩论
            "risk_debate_state": RiskDebateState(
                {
                    "history": "",                  # 完整辩论历史
                    "current_risky_response": "",   # 激进分析师的当前回应
                    "current_safe_response": "",    # 保守分析师的当前回应
                    "current_neutral_response": "", # 中性分析师的当前回应
                    "count": 0,                     # 辩论轮数计数器
                }
            ),  # 风险辩论状态，用于跟踪风险管理团队的讨论
            "market_report": "",         # 市场分析报告容器
            "fundamentals_report": "",   # 基本面分析报告容器
            "sentiment_report": "",      # 情绪分析报告容器
            "news_report": "",           # 新闻分析报告容器
        }

    def get_graph_args(self) -> Dict[str, Any]:
        """获取图执行的参数。
        
        Returns:
            包含图执行配置的字典
        """
        return {
            "stream_mode": "values",  # 流模式设置为值模式
            "config": {"recursion_limit": self.max_recur_limit},  # 设置递归限制
        }
