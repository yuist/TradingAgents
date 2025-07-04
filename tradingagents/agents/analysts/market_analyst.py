# 市场分析师模块
# 负责分析技术指标、价格走势和市场动态

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_market_analyst(llm, toolkit):
    """
    创建市场分析师节点
    
    市场分析师负责：
    1. 分析技术指标（移动平均线、MACD、RSI等）
    2. 评估市场趋势和动量
    3. 识别潜在的买卖信号
    4. 生成详细的市场分析报告
    
    Args:
        llm: 大语言模型实例
        toolkit: 工具包，包含各种数据获取工具
    
    Returns:
        market_analyst_node: 市场分析师节点函数
    """

    def market_analyst_node(state):
        # 获取当前交易日期和目标公司信息
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        # 根据配置选择在线或离线工具
        if toolkit.config["online_tools"]:
            # 在线工具：获取实时数据
            tools = [
                toolkit.get_YFin_data_online,  # Yahoo Finance 实时数据
                toolkit.get_stockstats_indicators_report_online,  # 技术指标实时报告
            ]
        else:
            # 离线工具：使用缓存数据
            tools = [
                toolkit.get_YFin_data,  # Yahoo Finance 历史数据
                toolkit.get_stockstats_indicators_report,  # 技术指标历史报告
            ]

        # 系统提示词
        system_message = (
            """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_YFin_data first to retrieve the CSV that is needed to generate indicators. Write a very detailed and nuanced report of the trends you observe. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."""
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
        )
        
        """
        系统提示词中文翻译：
        
        你是一个负责分析金融市场的交易助手。你的角色是从以下列表中选择与给定市场条件或交易策略**最相关的指标**。
        目标是选择最多**8个指标**，这些指标应该提供互补的见解而避免冗余。指标类别和每个类别的指标如下：

        移动平均线：
        - close_50_sma：50日简单移动平均线（SMA）：中期趋势指标。用途：识别趋势方向并作为动态支撑/阻力位。
          提示：存在价格滞后；结合更快的指标以获得及时的信号。
        - close_200_sma：200日SMA：长期趋势基准。用途：确认整体市场趋势并识别金叉/死叉形态。
          提示：反应缓慢；最适合用于战略性趋势确认而非频繁的交易入场。
        - close_10_ema：10日指数移动平均线（EMA）：反应灵敏的短期平均线。用途：捕捉动量的快速变化和潜在的入场点。
          提示：在震荡市场中容易产生噪音；与较长期平均线一起使用以过滤虚假信号。

        MACD相关指标：
        - macd：MACD（移动平均收敛散度）：通过EMA的差值计算动量。用途：寻找交叉和背离作为趋势变化的信号。
          提示：在低波动或横盘市场中需要与其他指标确认。
        - macds：MACD信号线：MACD线的EMA平滑。用途：使用与MACD线的交叉来触发交易。
          提示：应该作为更广泛策略的一部分以避免虚假信号。
        - macdh：MACD柱状图：显示MACD线与其信号线之间的差距。用途：可视化动量强度并尽早发现背离。
          提示：可能波动较大；在快速变动的市场中需要额外的过滤器。

        动量指标：
        - rsi：相对强弱指数（RSI）：衡量动量以标记超买/超卖状况。用途：应用70/30阈值并观察背离以预示反转。
          提示：在强趋势中，RSI可能保持极端值；始终与趋势分析交叉检查。

        波动率指标：
        - boll：布林带中线：作为布林带基础的20日SMA。用途：作为价格运动的动态基准。
          提示：与上下轨结合使用以有效发现突破或反转。
        - boll_ub：布林带上轨：通常在中线上方2个标准差。用途：信号潜在的超买状况和突破区域。
          提示：用其他工具确认信号；在强趋势中价格可能沿着轨道运行。
        - boll_lb：布林带下轨：通常在中线下方2个标准差。用途：指示潜在的超卖状况。
          提示：使用额外的分析以避免虚假的反转信号。
        - atr：平均真实波幅（ATR）：平均真实范围以衡量波动率。用途：设置止损水平并根据当前市场波动率调整仓位大小。
          提示：这是一个反应性指标，因此将其作为更广泛风险管理策略的一部分使用。

        基于成交量的指标：
        - vwma：成交量加权移动平均线（VWMA）：按成交量加权的移动平均线。用途：通过整合价格行动与成交量数据来确认趋势。
          提示：注意成交量激增导致的偏差结果；与其他成交量分析结合使用。

        - 选择提供多样化和互补信息的指标。避免冗余（例如，不要同时选择rsi和stochrsi）。
        - 还要简要解释为什么它们适合给定的市场环境。
        - 调用工具时，请使用上述提供的指标的确切名称，因为它们是定义的参数，否则调用将失败。
        - 请确保首先调用get_YFin_data来检索生成指标所需的CSV文件。
        - 撰写一份非常详细和细致的趋势观察报告。不要简单地说趋势是混合的，而要提供详细和精细的分析和见解，以帮助交易者做出决策。
        - 确保在报告末尾附加一个Markdown表格，以组织报告中的关键点，使其有条理且易于阅读。
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
                    "For your reference, the current date is {current_date}. The company we want to look at is {ticker}",
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
        供你参考，当前日期是{current_date}。我们要查看的公司是{ticker}。
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
            "messages": [result],  # 添加新消息到消息历史
            "market_report": report,  # 市场分析报告
        }

    return market_analyst_node
