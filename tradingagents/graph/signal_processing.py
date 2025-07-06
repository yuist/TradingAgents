# TradingAgents/graph/signal_processing.py
# 信号处理模块：处理交易信号并提取可操作的决策

from langchain_openai import ChatOpenAI


class SignalProcessor:
    """处理交易信号以提取可操作的决策。"""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """使用处理用的LLM初始化。
        
        Args:
            quick_thinking_llm: 用于快速处理的大语言模型实例
        """
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        处理完整的交易信号以提取核心决策。

        Args:
            full_signal: 完整的交易信号文本，通常是交易员的详细分析

        Returns:
            提取的决策（买入、卖出或持有）
        """
        messages = [
            (
                "system",
                "你是一个高效的助手，设计用于分析由分析师团队提供的段落或财务报告。你的任务是提取投资决策：卖出(SELL)、买入(BUY)或持有(HOLD)。在输出中只提供提取的决策(SELL、BUY或HOLD)，不要添加任何额外的文本或信息。",
            ),
            ("human", full_signal),
        ]

        return self.quick_thinking_llm.invoke(messages).content
