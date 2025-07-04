# 看跌研究员模块
# 负责构建反对投资股票的论据和风险分析

from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    """
    创建看跌研究员节点
    
    看跌研究员负责：
    1. 构建反对投资该股票的合理论据
    2. 强调风险、挑战和负面指标
    3. 反驳看涨研究员的论点
    4. 参与辩论以展示投资风险和弱点
    
    Args:
        llm: 大语言模型实例
        memory: 记忆存储，包含过去的经验和教训
    
    Returns:
        bear_node: 看跌研究员节点函数
    """
    
    def bear_node(state) -> dict:
        # 获取投资辩论状态
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")              # 完整辩论历史
        bear_history = investment_debate_state.get("bear_history", "")    # 看跌方历史

        # 获取当前回应和各种分析报告
        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]          # 市场分析报告
        sentiment_report = state["sentiment_report"]             # 情绪分析报告
        news_report = state["news_report"]                       # 新闻分析报告
        fundamentals_report = state["fundamentals_report"]       # 基本面分析报告

        # 整合当前市场情况
        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        
        # 从记忆中获取过去的经验
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        # 组织过去的记忆为字符串
        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        # 系统提示词
        prompt = f"""You are a Bear Analyst making the case against investing in the stock. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Resources available:

Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Reflections from similar situations and lessons learned: {past_memory_str}
Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the stock. You must also address reflections and learn from lessons and mistakes you made in the past.
"""
        
        """
        系统提示词中文翻译：
        
        你是一名看跌分析师，主张反对投资该股票。你的目标是提出一个合理的论证，
        强调风险、挑战和负面指标。利用提供的研究和数据来突出潜在的下行风险，
        并有效反驳看涨论点。

        关注要点：
        
        - 风险和挑战：突出可能阻碍股票表现的因素，如市场饱和、财务不稳定或宏观经济威胁。
        - 竞争劣势：强调弱势，如市场地位较弱、创新衰退或来自竞争对手的威胁。
        - 负面指标：使用来自财务数据、市场趋势或近期不利新闻的证据来支持你的立场。
        - 看涨反驳：用具体数据和合理推理批判性地分析看涨论点，暴露弱点或过度乐观的假设。
        - 参与性：以对话风格呈现你的论点，直接与看涨分析师的观点互动，
          进行有效辩论而不仅仅是列举事实。

        可用资源：

        市场研究报告：{market_research_report}
        社交媒体情绪报告：{sentiment_report}
        最新世界事务新闻：{news_report}
        公司基本面报告：{fundamentals_report}
        辩论的对话历史：{history}
        最后的看涨论点：{current_response}
        来自类似情况的反思和经验教训：{past_memory_str}
        
        使用这些信息提出令人信服的看跌论点，反驳看涨方的声明，
        并参与动态辩论，展示投资该股票的风险和弱点。
        你还必须处理反思并从过去的经验教训和错误中学习。
        """

        # 调用LLM生成看跌论点
        response = llm.invoke(prompt)

        # 格式化论点
        argument = f"Bear Analyst: {response.content}"

        # 更新投资辩论状态
        new_investment_debate_state = {
            "history": history + "\n" + argument,                      # 添加到完整历史
            "bear_history": bear_history + "\n" + argument,            # 添加到看跌历史
            "bull_history": investment_debate_state.get("bull_history", ""),  # 保持看涨历史
            "current_response": argument,                               # 设置当前回应
            "count": investment_debate_state["count"] + 1,              # 增加辩论轮数
        }

        # 返回更新后的状态
        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
