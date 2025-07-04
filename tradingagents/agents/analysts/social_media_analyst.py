# 社交媒体分析师模块
# 负责分析社交媒体舆情、公众情绪和市场情绪

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_social_media_analyst(llm, toolkit):
    """
    创建社交媒体分析师节点
    
    社交媒体分析师负责：
    1. 监控社交媒体平台（Reddit、Twitter等）的讨论
    2. 分析公众对特定公司的情绪和态度
    3. 识别市场情绪的转变和热点话题
    4. 生成详细的情绪分析报告
    
    Args:
        llm: 大语言模型实例
        toolkit: 工具包，包含各种社交媒体数据获取工具
    
    Returns:
        social_media_analyst_node: 社交媒体分析师节点函数
    """
    
    def social_media_analyst_node(state):
        # 获取当前交易日期和目标公司信息
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        # 根据配置选择在线或离线工具
        if toolkit.config["online_tools"]:
            # 在线工具：获取实时社交媒体数据
            tools = [toolkit.get_stock_news_openai]
        else:
            # 离线工具：使用缓存的社交媒体数据
            tools = [
                toolkit.get_reddit_stock_info,  # Reddit 股票讨论数据
            ]

        # 系统提示词
        system_message = (
            "You are a social media and company specific news researcher/analyst tasked with analyzing social media posts, recent company news, and public sentiment for a specific company over the past week. You will be given a company's name your objective is to write a comprehensive long report detailing your analysis, insights, and implications for traders and investors on this company's current state after looking at social media and what people are saying about that company, analyzing sentiment data of what people feel each day about the company, and looking at recent company news. Try to look at all sources possible from social media to sentiment to news. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
            + """ Make sure to append a Makrdown table at the end of the report to organize key points in the report, organized and easy to read.""",
        )
        
        """
        系统提示词中文翻译：
        
        你是一名社交媒体和公司特定新闻研究员/分析师，负责分析过去一周内特定公司的社交媒体帖子、最新公司新闻和公众情绪。
        你将获得一个公司名称，你的目标是撰写一份全面详尽的报告，详细说明你的分析、见解，
        以及对交易者和投资者的影响，内容基于：
        - 查看社交媒体和人们对该公司的评论
        - 分析人们每天对该公司的情绪数据
        - 查看最新的公司新闻
        
        尽量查看所有可能的来源，从社交媒体到情绪到新闻。
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
                    "For your reference, the current date is {current_date}. The current company we want to analyze is {ticker}",
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
        供你参考，当前日期是{current_date}。我们要分析的当前公司是{ticker}。
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
            "messages": [result],         # 添加新消息到消息历史
            "sentiment_report": report,   # 情绪分析报告
        }

    return social_media_analyst_node
