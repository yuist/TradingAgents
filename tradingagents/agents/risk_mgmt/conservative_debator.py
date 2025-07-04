# 保守派风险管理辩论者模块
# 负责从风险规避角度评估交易决策

from langchain_core.messages import AIMessage
import time
import json


def create_safe_debator(llm):
    """
    创建保守派风险分析师节点
    
    保守派风险分析师负责：
    1. 保护资产，最小化波动性，确保稳定可靠的增长
    2. 优先考虑稳定性、安全性和风险缓解
    3. 仔细评估潜在损失、经济衰退和市场波动
    4. 批判性地检查高风险要素，指出可能的风险暴露
    5. 与激进派和中立派辩论，强调保守策略的优势
    
    Args:
        llm: 大语言模型实例
    
    Returns:
        safe_node: 保守派风险分析师节点函数
    """
    
    def safe_node(state) -> dict:
        # 获取风险辩论状态
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")              # 完整辩论历史
        safe_history = risk_debate_state.get("safe_history", "")    # 保守派历史

        # 获取其他辩论者的回应
        current_risky_response = risk_debate_state.get("current_risky_response", "")      # 激进派回应
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")  # 中立派回应

        # 获取各种分析报告
        market_research_report = state["market_report"]         # 市场研究报告
        sentiment_report = state["sentiment_report"]            # 情绪分析报告
        news_report = state["news_report"]                      # 新闻分析报告
        fundamentals_report = state["fundamentals_report"]      # 基本面分析报告

        # 获取交易员的决策
        trader_decision = state["trader_investment_plan"]

        # 系统提示词
        prompt = f"""As the Safe/Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. When evaluating the trader's decision or plan, critically examine high-risk elements, pointing out where the decision may expose the firm to undue risk and where more cautious alternatives could secure long-term gains. Here is the trader's decision:

{trader_decision}

Your task is to actively counter the arguments of the Risky and Neutral Analysts, highlighting where their views may overlook potential threats or fail to prioritize sustainability. Respond directly to their points, drawing from the following data sources to build a convincing case for a low-risk approach adjustment to the trader's decision:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Here is the current conversation history: {history} Here is the last response from the risky analyst: {current_risky_response} Here is the last response from the neutral analyst: {current_neutral_response}. If there are no responses from the other viewpoints, do not halluncinate and just present your point.

Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked. Address each of their counterpoints to showcase why a conservative stance is ultimately the safest path for the firm's assets. Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy over their approaches. Output conversationally as if you are speaking without any special formatting."""

        """
        系统提示词中文翻译：
        
        作为保守派/安全风险分析师，你的主要目标是保护资产，最小化波动性，确保稳定可靠的增长。
        你优先考虑稳定性、安全性和风险缓解，仔细评估潜在损失、经济衰退和市场波动。
        当评估交易员的决策或计划时，批判性地检查高风险要素，指出决策可能使公司面临不必要风险的地方，
        以及更谨慎的替代方案如何确保长期收益。以下是交易员的决策：

        {trader_decision}

        你的任务是积极反驳激进派和中立派分析师的论点，突出他们的观点可能忽略潜在威胁或
        未能优先考虑可持续性的地方。直接回应他们的观点，利用以下数据源为交易员决策的
        低风险方法调整构建令人信服的案例：

        市场研究报告：{market_research_report}
        社交媒体情绪报告：{sentiment_report}
        最新世界事务报告：{news_report}
        公司基本面报告：{fundamentals_report}
        
        以下是当前对话历史：{history}
        以下是激进派分析师的最后回应：{current_risky_response}
        以下是中立派分析师的最后回应：{current_neutral_response}
        如果其他观点没有回应，不要虚构，只需提出你的观点。

        通过质疑他们的乐观主义并强调他们可能忽略的潜在不利因素来参与辩论。
        解决他们的每个反驳点，以展示为什么保守立场最终是公司资产最安全的路径。
        专注于辩论和批评他们的论点，以证明低风险策略相对于他们方法的优势。
        以对话方式输出，就像你在说话一样，不使用任何特殊格式。
        """

        # 调用LLM生成保守派论点
        response = llm.invoke(prompt)

        # 格式化论点
        argument = f"Safe Analyst: {response.content}"

        # 更新风险辩论状态
        new_risk_debate_state = {
            "history": history + "\n" + argument,                  # 添加到完整历史
            "risky_history": risk_debate_state.get("risky_history", ""),     # 保持激进派历史
            "safe_history": safe_history + "\n" + argument,        # 添加到保守派历史
            "neutral_history": risk_debate_state.get("neutral_history", ""), # 保持中立派历史
            "latest_speaker": "Safe",                              # 设置最新发言者
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),                                                     # 保持激进派回应
            "current_safe_response": argument,                     # 设置保守派回应
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),                                                     # 保持中立派回应
            "count": risk_debate_state["count"] + 1,               # 增加辩论轮数
        }

        # 返回更新后的状态
        return {"risk_debate_state": new_risk_debate_state}

    return safe_node
