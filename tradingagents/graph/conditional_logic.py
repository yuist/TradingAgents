# TradingAgents/graph/conditional_logic.py
# 条件逻辑模块：决定智能体图中的流程控制和转换条件

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """处理决定图流程的条件逻辑。"""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """使用配置参数初始化条件逻辑。
        
        Args:
            max_debate_rounds: 研究员之间最大辩论轮数
            max_risk_discuss_rounds: 风险管理团队最大讨论轮数
        """
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_market(self, state: AgentState):
        """确定市场分析是否应该继续。
        
        Args:
            state: 当前智能体状态
            
        Returns:
            下一步操作的名称
        """
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_market"  # 如果有工具调用，继续执行工具
        return "Msg Clear Market"  # 否则清除消息并进入下一阶段

    def should_continue_social(self, state: AgentState):
        """确定社交媒体分析是否应该继续。
        
        Args:
            state: 当前智能体状态
            
        Returns:
            下一步操作的名称
        """
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_social"  # 如果有工具调用，继续执行工具
        return "Msg Clear Social"  # 否则清除消息并进入下一阶段

    def should_continue_news(self, state: AgentState):
        """确定新闻分析是否应该继续。
        
        Args:
            state: 当前智能体状态
            
        Returns:
            下一步操作的名称
        """
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_news"  # 如果有工具调用，继续执行工具
        return "Msg Clear News"  # 否则清除消息并进入下一阶段

    def should_continue_fundamentals(self, state: AgentState):
        """确定基本面分析是否应该继续。
        
        Args:
            state: 当前智能体状态
            
        Returns:
            下一步操作的名称
        """
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_fundamentals"  # 如果有工具调用，继续执行工具
        return "Msg Clear Fundamentals"  # 否则清除消息并进入下一阶段

    def should_continue_debate(self, state: AgentState) -> str:
        """确定投资辩论是否应该继续。
        
        Args:
            state: 当前智能体状态
            
        Returns:
            下一个发言者的名称
        """

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 如果达到最大辩论轮数（2个智能体之间的来回交流）
            return "Research Manager"  # 交给研究经理做决策
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"  # 如果当前是看涨方发言，下一个轮到看跌方
        return "Bull Researcher"  # 否则轮到看涨方发言

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """确定风险分析是否应该继续。
        
        Args:
            state: 当前智能体状态
            
        Returns:
            下一个发言者的名称
        """
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 如果达到最大讨论轮数（3个智能体之间的来回交流）
            return "Risk Judge"  # 交给风险经理做决策
        if state["risk_debate_state"]["latest_speaker"].startswith("Risky"):
            return "Safe Analyst"  # 如果当前是激进分析师发言，下一个轮到保守分析师
        if state["risk_debate_state"]["latest_speaker"].startswith("Safe"):
            return "Neutral Analyst"  # 如果当前是保守分析师发言，下一个轮到中性分析师
        return "Risky Analyst"  # 否则轮到激进分析师发言
