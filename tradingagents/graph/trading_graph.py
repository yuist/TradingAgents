# TradingAgents/graph/trading_graph.py
# 交易智能体图的核心实现文件
# 该文件协调整个交易系统的运作流程

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

# 导入不同的 LLM 提供商接口
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph 工具节点，用于执行具体的数据获取任务
from langgraph.prebuilt import ToolNode

# 导入所有智能体模块
from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
# 金融记忆模块，用于存储历史决策和经验
from tradingagents.agents.utils.memory import FinancialSituationMemory
# 智能体状态定义
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.interface import set_config

# 导入图相关的各个组件
from .conditional_logic import ConditionalLogic  # 条件逻辑处理
from .setup import GraphSetup  # 图设置和配置
from .propagation import Propagator  # 信息传播器
from .reflection import Reflector  # 反思和学习模块
from .signal_processing import SignalProcessor  # 信号处理器


class TradingAgentsGraph:
    """
    主类：协调整个交易智能体框架
    
    该类负责：
    1. 初始化所有智能体（分析师、研究员、交易员、风险管理）
    2. 构建智能体之间的通信图
    3. 执行交易决策流程
    4. 处理反思和学习机制
    """

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
    ):
        """
        初始化交易智能体图和相关组件

        Args:
            selected_analysts: 要包含的分析师类型列表
                - market: 市场技术分析师（分析价格、成交量、技术指标）
                - social: 社交媒体情绪分析师（分析社交媒体舆情）
                - news: 新闻分析师（分析全球新闻和宏观经济指标）
                - fundamentals: 基本面分析师（分析公司财务和业绩）
            debug: 是否开启调试模式（会输出详细的执行过程）
            config: 配置字典。如果为 None，使用默认配置
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG

        # 更新数据流接口的配置
        set_config(self.config)

        # 创建必要的目录结构
        # 用于存储数据缓存
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # 初始化大语言模型（LLMs）
        # 根据配置选择不同的 LLM 提供商
        if self.config["llm_provider"].lower() == "openai" or self.config["llm_provider"] == "ollama" or self.config["llm_provider"] == "openrouter":
            # OpenAI 系列模型配置
            self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "anthropic":
            # Anthropic Claude 系列模型配置
            self.deep_thinking_llm = ChatAnthropic(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatAnthropic(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "google":
            # Google Gemini 系列模型配置
            self.deep_thinking_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"])
            self.quick_thinking_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"])
        else:
            raise ValueError(f"不支持的 LLM 提供商: {self.config['llm_provider']}")
        
        # 初始化工具包，包含所有数据获取和分析工具
        self.toolkit = Toolkit(config=self.config)

        # 初始化记忆存储
        # 每个关键智能体都有自己的记忆，用于存储历史经验
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)  # 看涨研究员记忆
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)  # 看跌研究员记忆
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)  # 交易员记忆
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)  # 投资裁判记忆
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)  # 风险管理记忆

        # 创建工具节点
        # 每个节点对应一种数据源或分析类型
        self.tool_nodes = self._create_tool_nodes()

        # 初始化各个组件
        self.conditional_logic = ConditionalLogic()  # 条件逻辑处理器
        self.graph_setup = GraphSetup(  # 图设置器
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.toolkit,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()  # 信息传播器
        self.reflector = Reflector(self.quick_thinking_llm)  # 反思器，用于事后学习
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)  # 信号处理器，解析交易决策

        # 状态跟踪
        self.curr_state = None  # 当前状态
        self.ticker = None  # 当前交易的股票代码
        self.log_states_dict = {}  # 日期到完整状态的映射

        # 设置智能体通信图
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """
        创建不同数据源的工具节点
        
        返回一个字典，包含各种类型的工具节点：
        - market: 市场数据工具（价格、技术指标等）
        - social: 社交媒体工具（Reddit、Twitter 等）
        - news: 新闻工具（全球新闻、公司新闻等）
        - fundamentals: 基本面工具（财务报表、内部交易等）
        """
        return {
            "market": ToolNode(
                [
                    # 在线工具（实时数据）
                    self.toolkit.get_YFin_data_online,  # Yahoo Finance 实时数据
                    self.toolkit.get_stockstats_indicators_report_online,  # 技术指标报告
                    # 离线工具（缓存数据）
                    self.toolkit.get_YFin_data,  # Yahoo Finance 历史数据
                    self.toolkit.get_stockstats_indicators_report,  # 技术指标历史报告
                ]
            ),
            "social": ToolNode(
                [
                    # 在线工具
                    self.toolkit.get_stock_news_openai,  # OpenAI 股票新闻分析
                    # 离线工具
                    self.toolkit.get_reddit_stock_info,  # Reddit 股票讨论数据
                ]
            ),
            "news": ToolNode(
                [
                    # 在线工具
                    self.toolkit.get_global_news_openai,  # OpenAI 全球新闻分析
                    self.toolkit.get_google_news,  # Google 新闻
                    # 离线工具
                    self.toolkit.get_finnhub_news,  # Finnhub 新闻数据
                    self.toolkit.get_reddit_news,  # Reddit 新闻讨论
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # 在线工具
                    self.toolkit.get_fundamentals_openai,  # OpenAI 基本面分析
                    # 离线工具
                    self.toolkit.get_finnhub_company_insider_sentiment,  # 内部人员情绪
                    self.toolkit.get_finnhub_company_insider_transactions,  # 内部交易数据
                    self.toolkit.get_simfin_balance_sheet,  # 资产负债表
                    self.toolkit.get_simfin_cashflow,  # 现金流量表
                    self.toolkit.get_simfin_income_stmt,  # 利润表
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """
        运行交易智能体图，为特定公司在特定日期生成交易决策
        
        执行流程：
        1. 分析师收集并分析数据
        2. 研究员进行辩论
        3. 交易员制定交易计划
        4. 风险管理团队评估风险
        5. 投资组合经理做出最终决策
        
        Args:
            company_name: 公司股票代码（如 "NVDA", "AAPL"）
            trade_date: 交易日期（格式：YYYY-MM-DD）
            
        Returns:
            (final_state, decision): 最终状态和处理后的交易决策
        """
        self.ticker = company_name

        # 初始化智能体状态
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # 调试模式：逐步跟踪执行过程
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    # 打印每个智能体的输出
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # 标准模式：直接执行
            final_state = self.graph.invoke(init_agent_state, **args)

        # 存储当前状态，用于后续反思
        self.curr_state = final_state

        # 记录状态到日志
        self._log_state(trade_date, final_state)

        # 返回决策和处理后的信号
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """
        将最终状态记录到 JSON 文件
        
        记录内容包括：
        - 各分析师的报告
        - 研究员辩论历史
        - 交易决策
        - 风险评估
        """
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],  # 目标公司
            "trade_date": final_state["trade_date"],  # 交易日期
            "market_report": final_state["market_report"],  # 市场分析报告
            "sentiment_report": final_state["sentiment_report"],  # 情绪分析报告
            "news_report": final_state["news_report"],  # 新闻分析报告
            "fundamentals_report": final_state["fundamentals_report"],  # 基本面分析报告
            "investment_debate_state": {  # 投资辩论状态
                "bull_history": final_state["investment_debate_state"]["bull_history"],  # 看涨方历史
                "bear_history": final_state["investment_debate_state"]["bear_history"],  # 看跌方历史
                "history": final_state["investment_debate_state"]["history"],  # 完整辩论历史
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],  # 当前回应
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],  # 裁判决定
            },
            "trader_investment_decision": final_state["trader_investment_plan"],  # 交易员投资决策
            "risk_debate_state": {  # 风险辩论状态
                "risky_history": final_state["risk_debate_state"]["risky_history"],  # 激进派历史
                "safe_history": final_state["risk_debate_state"]["safe_history"],  # 保守派历史
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],  # 中立派历史
                "history": final_state["risk_debate_state"]["history"],  # 完整历史
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],  # 裁判决定
            },
            "investment_plan": final_state["investment_plan"],  # 投资计划
            "final_trade_decision": final_state["final_trade_decision"],  # 最终交易决策
        }

        # 保存到文件
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """
        基于收益反思决策并更新记忆
        
        每个智能体都会：
        1. 分析自己的决策是否正确
        2. 总结经验教训
        3. 更新记忆以改进未来决策
        
        Args:
            returns_losses: 仓位收益（正数=盈利，负数=亏损）
        """
        # 看涨研究员反思
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        # 看跌研究员反思
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        # 交易员反思
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        # 投资裁判反思
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        # 风险管理反思
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """
        处理信号以提取核心决策
        
        将复杂的交易决策文本解析为简单的操作指令
        
        Args:
            full_signal: 完整的交易决策文本
            
        Returns:
            处理后的简化交易信号
        """
        return self.signal_processor.process_signal(full_signal)
