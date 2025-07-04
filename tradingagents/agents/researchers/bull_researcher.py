# 看涨研究员模块
# 负责构建支持买入股票的论据和分析

from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    """
    创建看涨研究员节点
    
    看涨研究员负责：
    1. 构建支持投资该股票的强有力论据
    2. 强调增长潜力、竞争优势和积极的市场指标
    3. 反驳看跌研究员的论点
    4. 参与辩论以证明看涨观点的优势
    
    Args:
        llm: 大语言模型实例
        memory: 记忆存储，包含过去的经验和教训
    
    Returns:
        bull_node: 看涨研究员节点函数
    """
    
    def bull_node(state) -> dict:
        # 获取投资辩论状态
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")              # 完整辩论历史
        bull_history = investment_debate_state.get("bull_history", "")    # 看涨方历史

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
        prompt = f"""You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Resources available:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Reflections from similar situations and lessons learned: {past_memory_str}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position. You must also address reflections and learn from lessons and mistakes you made in the past.
"""
        
        """
        系统提示词中文翻译：
        
        你是一名看涨分析师，主张投资该股票。你的任务是构建一个强有力的、基于证据的案例，
        强调增长潜力、竞争优势和积极的市场指标。利用提供的研究和数据来解决担忧并有效反驳看跌论点。

        关注要点：
        - 增长潜力：突出公司的市场机会、收入预测和可扩展性。
        - 竞争优势：强调独特产品、强势品牌或主导市场地位等因素。
        - 积极指标：将财务健康状况、行业趋势和近期积极新闻作为证据。
        - 看跌反驳：用具体数据和合理推理批判性地分析看跌论点，彻底解决担忧，
          并展示为什么看涨观点具有更强的优势。
        - 参与性：以对话风格呈现你的论点，直接与看跌分析师的观点互动，
          进行有效辩论而不仅仅是列举数据。

        可用资源：
        市场研究报告：{market_research_report}
        社交媒体情绪报告：{sentiment_report}
        最新世界事务新闻：{news_report}
        公司基本面报告：{fundamentals_report}
        辩论的对话历史：{history}
        最后的看跌论点：{current_response}
        来自类似情况的反思和经验教训：{past_memory_str}
        
        使用这些信息提出令人信服的看涨论点，反驳看跌方的担忧，
        并参与动态辩论，展示看涨立场的优势。
        你还必须处理反思并从过去的经验教训和错误中学习。
        """

        # 调用LLM生成看涨论点
        response = llm.invoke(prompt)

        # 格式化论点
        argument = f"Bull Analyst: {response.content}"

        # 更新投资辩论状态
        new_investment_debate_state = {
            "history": history + "\n" + argument,                      # 添加到完整历史
            "bull_history": bull_history + "\n" + argument,            # 添加到看涨历史
            "bear_history": investment_debate_state.get("bear_history", ""),  # 保持看跌历史
            "current_response": argument,                               # 设置当前回应
            "count": investment_debate_state["count"] + 1,              # 增加辩论轮数
        }

        # 返回更新后的状态
        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
