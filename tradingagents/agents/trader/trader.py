# 交易员模块
# 负责整合分析报告并制定具体的交易决策

import functools
import time
import json


def create_trader(llm, memory):
    """
    创建交易员节点
    
    交易员负责：
    1. 整合所有分析师的报告和研究员的辩论结果
    2. 考虑技术分析、基本面分析、情绪分析和新闻分析
    3. 利用历史经验和记忆进行决策
    4. 提供明确的交易建议（买入/持有/卖出）
    
    Args:
        llm: 大语言模型实例
        memory: 记忆存储，包含过去的交易经验和教训
    
    Returns:
        trader_node: 部分应用的交易员节点函数
    """
    
    def trader_node(state, name):
        # 获取公司信息和投资计划
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]              # 研究员辩论后的投资计划
        
        # 获取各分析师的报告
        market_research_report = state["market_report"]         # 市场技术分析报告
        sentiment_report = state["sentiment_report"]            # 社交媒体情绪报告
        news_report = state["news_report"]                      # 新闻分析报告
        fundamentals_report = state["fundamentals_report"]      # 基本面分析报告

        # 整合当前市场情况
        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        
        # 从记忆中获取过去的交易经验
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        # 组织过去的记忆为字符串
        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        # 构建上下文信息
        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }
        
        """
        上下文信息中文翻译：
        
        基于分析师团队的全面分析，这里有一个为{company_name}量身定制的投资计划。
        该计划结合了当前技术市场趋势、宏观经济指标和社交媒体情绪的见解。
        将此计划作为评估你下一个交易决策的基础。
        
        建议的投资计划：{investment_plan}
        
        利用这些见解做出明智和战略性的决策。
        """

        # 构建消息列表
        messages = [
            {
                "role": "system",
                "content": f"""You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold. End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation. Do not forget to utilize lessons from past decisions to learn from your mistakes. Here is some reflections from similar situatiosn you traded in and the lessons learned: {past_memory_str}""",
            },
            context,
        ]
        
        """
        系统消息中文翻译：
        
        你是一个分析市场数据以做出投资决策的交易智能体。
        基于你的分析，提供买入、卖出或持有的具体建议。
        以坚定的决策结束，并始终以"最终交易提案：**买入/持有/卖出**"来结束你的回应，以确认你的建议。
        不要忘记利用过去决策的经验教训来从错误中学习。
        这里是你在类似情况下交易的一些反思和学到的经验教训：{past_memory_str}
        """

        # 调用LLM生成交易决策
        result = llm.invoke(messages)

        # 返回交易决策结果
        return {
            "messages": [result],                      # 消息历史
            "trader_investment_plan": result.content,  # 交易员的投资计划
            "sender": name,                            # 发送者名称
        }

    # 返回部分应用的函数，固定name参数为"Trader"
    return functools.partial(trader_node, name="Trader")
