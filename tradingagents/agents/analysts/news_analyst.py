# 新闻分析师模块
# 负责分析全球新闻、宏观经济事件和市场影响

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_news_analyst(llm, toolkit):
    """
    创建新闻分析师节点
    
    新闻分析师负责：
    1. 监控全球新闻和宏观经济指标
    2. 分析新闻事件对市场的潜在影响
    3. 识别可能影响股价的重大事件
    4. 生成详细的新闻分析报告
    
    Args:
        llm: 大语言模型实例
        toolkit: 工具包，包含各种新闻数据获取工具
    
    Returns:
        news_analyst_node: 新闻分析师节点函数
    """
    
    def news_analyst_node(state):
        # 获取当前交易日期和目标公司信息
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        # 根据配置选择在线或离线工具
        if toolkit.config["online_tools"]:
            # 在线工具：获取实时新闻
            tools = [toolkit.get_global_news_openai, toolkit.get_google_news]
        else:
            # 离线工具：使用缓存的新闻数据
            tools = [
                toolkit.get_finnhub_news,  # Finnhub 新闻源
                toolkit.get_reddit_news,   # Reddit 新闻讨论
                toolkit.get_google_news,   # Google 新闻
            ]

        # 系统提示词
        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Look at news from EODHD, and finnhub to be comprehensive. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
            + """ Make sure to append a Makrdown table at the end of the report to organize key points in the report, organized and easy to read."""
        )
        
        """
        系统提示词中文翻译：
        
        你是一名新闻研究员，负责分析过去一周的最新新闻和趋势。
        请撰写一份全面的报告，介绍与交易和宏观经济相关的世界现状。
        查看来自 EODHD 和 finnhub 的新闻以确保全面性。
        不要简单地说趋势是混合的，而要提供详细和精细的分析和见解，以帮助交易者做出决策。
        确保在报告末尾附加一个 Markdown 表格，以组织报告中的关键点，使其有条理且易于阅读。
        """

        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    # 系统角色定义
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. We are looking at the company {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        """
        系统角色定义中文翻译：
        
        你是一个有帮助的AI助手，与其他助手协作。
        使用提供的工具来推进回答问题的进度。
        如果你无法完全回答，没关系；另一个具有不同工具的助手会从你停下的地方继续。
        执行你能做的部分以取得进展。
        如果你或任何其他助手有最终交易提案：**买入/持有/卖出**或可交付成果，
        请在你的响应前加上"最终交易提案：**买入/持有/卖出**"，以便团队知道何时停止。
        你可以访问以下工具：{tool_names}。
        {system_message}
        供你参考，当前日期是{current_date}。我们正在查看公司{ticker}。
        """

        # 填充提示模板的参数
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        # 创建执行链：提示 -> LLM（绑定工具）
        chain = prompt | llm.bind_tools(tools)
        
        # 执行分析
        result = chain.invoke(state["messages"])

        # 初始化报告
        report = ""

        # 如果没有工具调用，直接使用返回的内容作为报告
        if len(result.tool_calls) == 0:
            report = result.content

        # 返回更新后的状态
        return {
            "messages": [result],    # 添加新消息到消息历史
            "news_report": report,   # 新闻分析报告
        }

    return news_analyst_node
