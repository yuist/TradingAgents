# 风险管理经理模块
# 负责评估风险辩论并做出最终的风险管理决策

import time
import json


def create_risk_manager(llm, memory):
    """
    创建风险管理经理节点
    
    风险管理经理负责：
    1. 评估激进派、中立派和保守派三方的风险辩论
    2. 确定最佳的行动方案
    3. 提供明确的推荐：买入、卖出或持有
    4. 基于辩论和历史经验完善交易员的计划
    5. 从过去的错误中学习，避免重复亏损决策
    
    Args:
        llm: 大语言模型实例
        memory: 记忆存储，包含过去的风险管理经验
    
    Returns:
        risk_manager_node: 风险管理经理节点函数
    """
    
    def risk_manager_node(state) -> dict:
        # 获取基本信息
        company_name = state["company_of_interest"]

        # 获取风险辩论相关信息
        history = state["risk_debate_state"]["history"]          # 风险辩论历史
        risk_debate_state = state["risk_debate_state"]           # 风险辩论状态
        
        # 获取各种分析报告
        market_research_report = state["market_report"]         # 市场研究报告
        news_report = state["news_report"]                      # 新闻分析报告
        fundamentals_report = state["news_report"]              # 基本面分析报告（注意：这里似乎有错误，应该是fundamentals_report）
        sentiment_report = state["sentiment_report"]            # 情绪分析报告
        trader_plan = state["investment_plan"]                  # 交易员投资计划

        # 整合当前市场情况
        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        
        # 从记忆中获取过去的经验
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        # 组织过去的记忆为字符串
        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        # 系统提示词
        prompt = f"""As the Risk Management Judge and Debate Facilitator, your goal is to evaluate the debate between three risk analysts—Risky, Neutral, and Safe/Conservative—and determine the best course of action for the trader. Your decision must result in a clear recommendation: Buy, Sell, or Hold. Choose Hold only if strongly justified by specific arguments, not as a fallback when all sides seem valid. Strive for clarity and decisiveness.

Guidelines for Decision-Making:
1. **Summarize Key Arguments**: Extract the strongest points from each analyst, focusing on relevance to the context.
2. **Provide Rationale**: Support your recommendation with direct quotes and counterarguments from the debate.
3. **Refine the Trader's Plan**: Start with the trader's original plan, **{trader_plan}**, and adjust it based on the analysts' insights.
4. **Learn from Past Mistakes**: Use lessons from **{past_memory_str}** to address prior misjudgments and improve the decision you are making now to make sure you don't make a wrong BUY/SELL/HOLD call that loses money.

Deliverables:
- A clear and actionable recommendation: Buy, Sell, or Hold.
- Detailed reasoning anchored in the debate and past reflections.

---

**Analysts Debate History:**  
{history}

---

Focus on actionable insights and continuous improvement. Build on past lessons, critically evaluate all perspectives, and ensure each decision advances better outcomes."""

        """
        系统提示词中文翻译：
        
        作为风险管理裁判和辩论促进者，你的目标是评估三位风险分析师——激进派、中立派和保守派——之间的辩论，
        并确定交易员的最佳行动方案。你的决策必须产生明确的推荐：买入、卖出或持有。
        只有在有具体论据强力支持时才选择持有，而不是在所有方面都看似有效时的后备选择。
        努力保持清晰和决断性。

        决策指导原则：
        1. **总结关键论点**：从每位分析师那里提取最强的观点，专注于与背景相关的内容。
        2. **提供理由**：用辩论中的直接引用和反驳论点支持你的推荐。
        3. **完善交易员计划**：从交易员的原始计划**{trader_plan}**开始，基于分析师的见解进行调整。
        4. **从过去的错误中学习**：使用**{past_memory_str}**中的经验教训来解决先前的误判，
           改进你现在做出的决策，确保不会做出错误的买入/卖出/持有决定而亏损。

        交付物：
        - 明确且可执行的推荐：买入、卖出或持有。
        - 基于辩论和过去反思的详细推理。

        ---

        **分析师辩论历史：**  
        {history}

        ---

        专注于可执行的见解和持续改进。基于过去的经验教训，批判性地评估所有观点，
        确保每个决策都能推进更好的结果。
        """

        # 调用LLM生成风险管理决策
        response = llm.invoke(prompt)

        # 更新风险辩论状态，添加裁判决定
        new_risk_debate_state = {
            "judge_decision": response.content,                     # 裁判决定
            "history": risk_debate_state["history"],               # 保持辩论历史
            "risky_history": risk_debate_state["risky_history"],   # 保持激进派历史
            "safe_history": risk_debate_state["safe_history"],     # 保持保守派历史
            "neutral_history": risk_debate_state["neutral_history"], # 保持中立派历史
            "latest_speaker": "Judge",                             # 设置最新发言者为裁判
            "current_risky_response": risk_debate_state["current_risky_response"],   # 保持激进派回应
            "current_safe_response": risk_debate_state["current_safe_response"],     # 保持保守派回应
            "current_neutral_response": risk_debate_state["current_neutral_response"], # 保持中立派回应
            "count": risk_debate_state["count"],                   # 保持辩论轮数
        }

        # 返回更新后的状态和最终交易决策
        return {
            "risk_debate_state": new_risk_debate_state,            # 更新的风险辩论状态
            "final_trade_decision": response.content,              # 最终交易决策
        }

    return risk_manager_node
