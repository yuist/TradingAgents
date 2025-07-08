from typing import Optional
import datetime
import typer
from pathlib import Path
from functools import wraps
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.live import Live
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule
import sys
import os

# 导入优化的显示模块
from .optimized_display import (
    OptimizedDisplay, DisplayConfig, DisplayMode, MessageType,
    get_display, set_display_mode
)

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.config_manager import get_config_manager
from tradingagents.utils.logging_manager import LoggerManager
import os
import logging
import logging.handlers
from .models import AnalystType
from .utils import *

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)

# 全局显示配置
GLOBAL_DISPLAY_CONFIG = DisplayConfig()
GLOBAL_OPTIMIZED_DISPLAY = None


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {
            # Analyst Team
            "Market Analyst": "pending",
            "Social Analyst": "pending",
            "News Analyst": "pending",
            "Fundamentals Analyst": "pending",
            # Research Team
            "Bull Researcher": "pending",
            "Bear Researcher": "pending",
            "Research Manager": "pending",
            # Trading Team
            "Trader": "pending",
            # Risk Management Team
            "Risky Analyst": "pending",
            "Neutral Analyst": "pending",
            "Safe Analyst": "pending",
            # Portfolio Management Team
            "Portfolio Manager": "pending",
        }
        self.current_agent = None
        self.report_sections = {
            "market_report": None,
            "sentiment_report": None,
            "news_report": None,
            "fundamentals_report": None,
            "investment_plan": None,
            "trader_investment_plan": None,
            "final_trade_decision": None,
        }

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content
               
        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports
        if any(
            self.report_sections[section]
            for section in [
                "market_report",
                "sentiment_report",
                "news_report",
                "fundamentals_report",
            ]
        ):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections["market_report"]:
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections["sentiment_report"]:
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections["news_report"]:
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections["fundamentals_report"]:
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # Research Team Reports
        if self.report_sections["investment_plan"]:
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections["trader_investment_plan"]:
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections["final_trade_decision"]:
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def update_display(layout, spinner_text=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team
    teams = {
        "Analyst Team": [
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Risky Analyst", "Neutral Analyst", "Safe Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status[first_agent]
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status[agent]
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        # Truncate tool call args if too long
        if isinstance(args, str) and len(args) > 100:
            args = args[:97] + "..."
        all_messages.append((timestamp, "Tool", f"{tool_name}: {args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        # Convert content to string if it's not already
        content_str = content
        if isinstance(content, list):
            # Handle list of content blocks (Anthropic format)
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'tool_use':
                        text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
                else:
                    text_parts.append(str(item))
            content_str = ' '.join(text_parts)
        elif not isinstance(content_str, str):
            content_str = str(content)
            
        # Truncate message content if too long
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp
    all_messages.sort(key=lambda x: x[0])

    # Calculate how many messages we can show based on available space
    # Start with a reasonable number and adjust based on content length
    max_messages = 12  # Increased from 8 to better fill the space

    # Get the last N messages that will fit in the panel
    recent_messages = all_messages[-max_messages:]

    # Add messages to table
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    if spinner_text:
        messages_table.add_row("", "Spinner", spinner_text)

    # Add a footer to indicate if messages were truncated
    if len(all_messages) > max_messages:
        messages_table.footer = (
            f"[dim]Showing last {max_messages} of {len(all_messages)} messages[/dim]"
        )

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    tool_calls_count = len(message_buffer.tool_calls)
    llm_calls_count = sum(
        1 for _, msg_type, _ in message_buffer.messages if msg_type == "Reasoning"
    )
    reports_count = sum(
        1 for content in message_buffer.report_sections.values() if content is not None
    )

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(
        f"Tool Calls: {tool_calls_count} | LLM Calls: {llm_calls_count} | Generated Reports: {reports_count}"
    )

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections():
    """Get all user selections before starting the analysis display."""
    # Display ASCII art welcome message
    with open("./cli/static/welcome.txt", "r", encoding="utf-8") as f:
        welcome_ascii = f.read()
        
    # 检查是否存在配置文件
    config_path = "config.yaml"
    config_exists = os.path.exists(config_path)
    config_manager = None
    llm_config = None
    
    if config_exists:
        # 加载配置文件以获取默认值
        from tradingagents.config_manager import ConfigManager
        config_manager = ConfigManager(config_path)
        llm_config = config_manager.get_llm_config()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()  # Add a blank line after the welcome box

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol", "Enter the ticker symbol to analyze", "SPY"
        )
    )
    selected_ticker = get_ticker()

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Select analysts
    console.print(
        create_question_box(
            "Step 3: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts()
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 4: Research depth
    console.print(
        create_question_box(
            "Step 4: Research Depth", "Select your research depth level"
        )
    )
    selected_research_depth = select_research_depth()
    
    # 如果存在配置文件，直接使用配置文件中的LLM设置
    if config_exists and llm_config:
        # 直接使用配置文件中的LLM设置
        console.print(
            create_question_box(
                "LLM配置", 
                f"使用配置文件中的LLM设置:\n提供商: {llm_config.provider}\n深度思考模型: {llm_config.deep_think_model}\n快速思考模型: {llm_config.quick_think_model}"
            )
        )
        
        console.print(f"[green]使用配置文件中的LLM设置:[/green]")
        console.print(f"[green]- 提供商:[/green] {llm_config.provider}")
        console.print(f"[green]- 深度思考模型:[/green] {llm_config.deep_think_model}")
        console.print(f"[green]- 快速思考模型:[/green] {llm_config.quick_think_model}")
        
        # 构建返回结果（不包含LLM相关键，直接使用配置文件）
        result = {
            "ticker": selected_ticker,
            "analysis_date": analysis_date,
            "analysts": selected_analysts,
            "research_depth": selected_research_depth,
        }
    else:
        # 没有配置文件时，手动选择LLM设置
        # Step 5: LLM提供商
        console.print(
            create_question_box(
                "Step 5: LLM提供商", "选择要使用的LLM服务"
            )
        )
        selected_llm_provider, backend_url = select_llm_provider()
        
        # Step 6: 思考代理
        console.print(
            create_question_box(
                "Step 6: 思考代理", "选择用于分析的思考代理"
            )
        )
        selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
        selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)
        
        # 构建返回结果（包含手动选择的LLM设置）
        result = {
            "ticker": selected_ticker,
            "analysis_date": analysis_date,
            "analysts": selected_analysts,
            "research_depth": selected_research_depth,
            "llm_provider": selected_llm_provider.lower(),
            "backend_url": backend_url,
            "shallow_thinker": selected_shallow_thinker,
            "deep_thinker": selected_deep_thinker,
        }
    
    return result


def normalize_ticker(ticker):
    """标准化股票代码格式，避免重复文件夹问题。
    
    Args:
        ticker: 原始股票代码
        
    Returns:
        标准化后的股票代码
    """
    ticker = ticker.upper().strip()
    
    # 检测中国股票代码（6位数字）
    import re
    if re.match(r'^\d{6}$', ticker):
        # 纯6位数字，直接返回（不添加后缀）
        return ticker
    elif re.match(r'^\d{6}\.(SZ|sz|SH|sh)$', ticker):
        # 6位数字+交易所后缀，移除后缀统一格式
        return ticker[:6]
    else:
        # 其他格式（如美股），直接返回
        return ticker


def get_ticker():
    """Get ticker symbol from user input."""
    import re
    while True:
        ticker = typer.prompt("", default="SPY")
        # 验证ticker格式：只允许字母、数字、点号和连字符
        if re.match(r'^[A-Za-z0-9.-]+$', ticker):
            # 标准化股票代码格式
            normalized_ticker = normalize_ticker(ticker)
            if normalized_ticker != ticker.upper():
                console.print(f"[yellow]Info: Ticker standardized from '{ticker}' to '{normalized_ticker}'[/yellow]")
            return normalized_ticker
        else:
            console.print("[red]Error: Invalid ticker symbol. Please use only letters, numbers, dots, and hyphens.[/red]")


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def display_complete_report(final_state):
    """Display the complete analysis report with team-based panels."""
    console.print("\n[bold green]Complete Analysis Report[/bold green]\n")

    # I. Analyst Team Reports
    analyst_reports = []

    # Market Analyst Report
    if final_state.get("market_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["market_report"]),
                title="Market Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Social Analyst Report
    if final_state.get("sentiment_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["sentiment_report"]),
                title="Social Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # News Analyst Report
    if final_state.get("news_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["news_report"]),
                title="News Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Fundamentals Analyst Report
    if final_state.get("fundamentals_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["fundamentals_report"]),
                title="Fundamentals Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    if analyst_reports:
        console.print(
            Panel(
                Columns(analyst_reports, equal=True, expand=True),
                title="I. Analyst Team Reports",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        research_reports = []
        debate_state = final_state["investment_debate_state"]

        # Bull Researcher Analysis
        if debate_state.get("bull_history"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["bull_history"]),
                    title="Bull Researcher",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Bear Researcher Analysis
        if debate_state.get("bear_history"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["bear_history"]),
                    title="Bear Researcher",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Research Manager Decision
        if debate_state.get("judge_decision"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["judge_decision"]),
                    title="Research Manager",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if research_reports:
            console.print(
                Panel(
                    Columns(research_reports, equal=True, expand=True),
                    title="II. Research Team Decision",
                    border_style="magenta",
                    padding=(1, 2),
                )
            )

    # III. Trading Team Reports
    if final_state.get("trader_investment_plan"):
        console.print(
            Panel(
                Panel(
                    Markdown(final_state["trader_investment_plan"]),
                    title="Trader",
                    border_style="blue",
                    padding=(1, 2),
                ),
                title="III. Trading Team Plan",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    # IV. Risk Management Team Reports
    if final_state.get("risk_debate_state"):
        risk_reports = []
        risk_state = final_state["risk_debate_state"]

        # Aggressive (Risky) Analyst Analysis
        if risk_state.get("risky_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["risky_history"]),
                    title="Aggressive Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Conservative (Safe) Analyst Analysis
        if risk_state.get("safe_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["safe_history"]),
                    title="Conservative Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Neutral Analyst Analysis
        if risk_state.get("neutral_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["neutral_history"]),
                    title="Neutral Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if risk_reports:
            console.print(
                Panel(
                    Columns(risk_reports, equal=True, expand=True),
                    title="IV. Risk Management Team Decision",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # V. Portfolio Manager Decision
        if risk_state.get("judge_decision"):
            console.print(
                Panel(
                    Panel(
                        Markdown(risk_state["judge_decision"]),
                        title="Portfolio Manager",
                        border_style="blue",
                        padding=(1, 2),
                    ),
                    title="V. Portfolio Manager Decision",
                    border_style="green",
                    padding=(1, 2),
                )
            )


def update_research_team_status(status):
    """Update status for all research team members and trader."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager", "Trader"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)

def extract_content_string(content):
    """Extract string content from various message formats."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle Anthropic's list format
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif item.get('type') == 'tool_use':
                    text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
            else:
                text_parts.append(str(item))
        return ' '.join(text_parts)
    else:
        return str(content)

def run_analysis():
    # First get all user selections
    selections = get_user_selections()

    # 检查是否存在配置文件
    config_path = "config.yaml"
    if os.path.exists(config_path):
        # 使用配置文件初始化
        from tradingagents.config_manager import ConfigManager
        config_manager = ConfigManager(config_path)
        
        # 从配置文件获取基础配置
        config = DEFAULT_CONFIG.copy()
        
        # 使用用户选择覆盖配置文件中的设置
        config["max_debate_rounds"] = selections["research_depth"]
        config["max_risk_discuss_rounds"] = selections["research_depth"]
        
        # 如果用户手动选择了LLM设置，则覆盖配置文件中的设置
        # 这些值会通过TradingAgentsGraph的初始化传递给ConfigManager
        temp_config = None
        if "llm_provider" in selections and "shallow_thinker" in selections and "deep_thinker" in selections:
            temp_config = config.copy()
            temp_config["llm_provider"] = selections["llm_provider"].lower()
            temp_config["quick_think_llm"] = selections["shallow_thinker"]
            temp_config["deep_think_llm"] = selections["deep_thinker"]
            temp_config["backend_url"] = selections["backend_url"]
    else:
        # 传统配置方式（向后兼容）
        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = selections["research_depth"]
        config["max_risk_discuss_rounds"] = selections["research_depth"]
        config["quick_think_llm"] = selections["shallow_thinker"]
        config["deep_think_llm"] = selections["deep_thinker"]
        config["backend_url"] = selections["backend_url"]
        config["llm_provider"] = selections["llm_provider"].lower()
        temp_config = None

    # Create result directory first (before using results_dir)
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Now initialize the graph with the correct debug_log_file path
    debug_log_file = results_dir / "debug_messages.log"
    
    if os.path.exists(config_path):
        # 初始化图（使用配置文件）
        graph = TradingAgentsGraph(
            selected_analysts=[analyst.value for analyst in selections["analysts"]],
            debug=True,
            config_path=config_path,
            config=temp_config,  # 如果用户手动选择了LLM，则传递临时配置
            debug_log_file=str(debug_log_file)
        )
    else:
        # 初始化图（使用传统配置）
        graph = TradingAgentsGraph(
            selected_analysts=[analyst.value for analyst in selections["analysts"]],
            debug=True,
            config=config,
            debug_log_file=str(debug_log_file)
        )

    # Create additional directories
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)
    
    # 初始化统一日志管理器
    config_manager = get_config_manager()
    logging_config = config_manager.get_logging_config()
    logger_manager = LoggerManager(logging_config)
    
    # 获取CLI专用日志器
    cli_logger = logger_manager.get_logger('cli')
    
    # 创建调试消息记录函数
    def log_debug_message(msg_type, content, agent_name=None):
        """记录详细的调试消息"""
        # 格式化消息内容
        if isinstance(content, list):
            # 处理复杂的消息格式
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'tool_use':
                        text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
                else:
                    text_parts.append(str(item))
            content_str = ' '.join(text_parts)
        else:
            content_str = str(content)
        
        # 使用统一日志管理器记录到文件（不输出到控制台）
        extra_data = {
            'message_type': msg_type,
            'agent_name': agent_name,
            'content_length': len(content_str)
        }
        
        if agent_name:
            extra_data['agent'] = agent_name
        
        # 只记录到文件，不输出到控制台
        # 创建一个临时的文件专用日志器
        file_logger = logging.getLogger(f'tradingagents.cli.debug')
        file_logger.setLevel(logging.DEBUG)
        
        # 确保只有文件处理器，没有控制台处理器
        if not file_logger.handlers:
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / "debug.log",
                maxBytes=50 * 1024 * 1024,
                backupCount=5,
                encoding='utf-8'
            )
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            file_logger.addHandler(file_handler)
            file_logger.propagate = False  # 防止传播到根日志器
        
        # 根据消息类型选择日志级别，但只记录到文件
        if msg_type == "System":
            file_logger.info(f"[{msg_type}] {content_str}", extra=extra_data)
        elif msg_type == "Reasoning":
            file_logger.debug(f"[{msg_type}] {content_str}", extra=extra_data)
        elif msg_type == "Tool":
            file_logger.info(f"[{msg_type}] {content_str}", extra=extra_data)
        else:
            file_logger.debug(f"[{msg_type}] {content_str}", extra=extra_data)
        
        # Debug消息只记录到文件，不显示到界面以避免刷屏
        # 如果需要查看debug信息，可以查看日志文件

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            
            # 使用统一日志管理器记录消息
            extra_data = {
                'timestamp': timestamp,
                'message_type': message_type,
                'source': 'cli_message_buffer'
            }
            
            # 记录到详细调试日志
            log_debug_message(message_type, content)
            
            # 同时记录到传统日志文件以保持兼容性
            content_simple = content.replace("\n", " ") if isinstance(content, str) else str(content)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content_simple}\n")
        return wrapper
    
    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, tool_args = obj.tool_calls[-1]
            
            # 使用统一日志管理器记录工具调用
            extra_data = {
                'timestamp': timestamp,
                'tool_name': tool_name,
                'tool_args': tool_args,
                'source': 'cli_tool_call'
            }
            
            args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items()) if isinstance(tool_args, dict) else str(tool_args)
            cli_logger.info(f"Tool Call: {tool_name}({args_str})", extra=extra_data)
            
            # 同时记录到传统日志文件以保持兼容性
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    with open(report_dir / file_name, "w", encoding="utf-8") as f:
                        f.write(content)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # 使用优化的显示器
    global GLOBAL_OPTIMIZED_DISPLAY
    display = GLOBAL_OPTIMIZED_DISPLAY
    
    # 如果是静默模式，使用传统的简单输出
    if display.config.mode == DisplayMode.SILENT:
        console.print(f"[green]开始分析 {selections['ticker']} ({selections['analysis_date']})[/green]")
        console.print(f"[blue]选择的分析师: {', '.join(analyst.value for analyst in selections['analysts'])}[/blue]")
    else:
        # 启动优化显示器
        display.start()
        
        # 添加初始消息
        display.log_system(f"选择股票代码: {selections['ticker']}")
        display.log_system(f"分析日期: {selections['analysis_date']}")
        display.log_system(f"选择的分析师: {', '.join(analyst.value for analyst in selections['analysts'])}")

    # 重置代理状态
    agent_names = [
        "Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst",
        "Bull Researcher", "Bear Researcher", "Research Manager",
        "Trader",
        "Risky Analyst", "Neutral Analyst", "Safe Analyst",
        "Portfolio Manager"
    ]
    
    for agent in agent_names:
        if display.config.mode != DisplayMode.SILENT:
            display.update_agent_status(agent, "pending")
        else:
            message_buffer.update_agent_status(agent, "pending")
    
    # 设置第一个分析师为进行中
    first_analyst = f"{selections['analysts'][0].value.capitalize()} Analyst"
    if display.config.mode != DisplayMode.SILENT:
        display.update_agent_status(first_analyst, "in_progress")
        display.log_system(f"开始分析 {selections['ticker']} ({selections['analysis_date']})...")
    else:
        message_buffer.update_agent_status(first_analyst, "in_progress")
        console.print(f"[yellow]启动 {first_analyst}...[/yellow]")

    # 初始化状态和图参数
    init_agent_state = graph.propagator.create_initial_state(
        selections["ticker"], selections["analysis_date"]
    )
    args = graph.propagator.get_graph_args()

    # 开始流式分析
    trace = []
    try:
        for chunk in graph.graph.stream(init_agent_state, **args):
            # 记录chunk的详细信息到调试日志
            log_debug_message("Chunk", f"Processing chunk with keys: {list(chunk.keys())}")
            
            if len(chunk["messages"]) > 0:
                # Get the last message from the chunk
                last_message = chunk["messages"][-1]

                # Extract message content and type
                if hasattr(last_message, "content"):
                    content = extract_content_string(last_message.content)  # Use the helper function
                    msg_type = "Reasoning"
                else:
                    content = str(last_message)
                    msg_type = "System"

                # 记录原始消息到调试日志
                log_debug_message(f"Raw_{msg_type}", last_message)

                # 根据显示模式处理消息
                if display.config.mode == DisplayMode.SILENT:
                    # 静默模式：只记录到buffer
                    message_buffer.add_message(msg_type, content)
                else:
                    # 使用优化显示器记录消息
                    if msg_type == "Reasoning":
                        display.log_agent("AI Agent", content)
                    else:
                        display.log_system(content)
                    # 同时保持原有buffer用于报告生成
                    message_buffer.add_message(msg_type, content)

                # If it's a tool call, add it to tool calls
                if hasattr(last_message, "tool_calls"):
                    for tool_call in last_message.tool_calls:
                        # Handle both dictionary and object tool calls
                        if isinstance(tool_call, dict):
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                        else:
                            tool_name = tool_call.name
                            tool_args = tool_call.args
                        
                        # 根据显示模式处理工具调用
                        if display.config.mode == DisplayMode.SILENT:
                            message_buffer.add_tool_call(tool_name, tool_args)
                        else:
                            display.log_tool(tool_name, tool_args)
                            message_buffer.add_tool_call(tool_name, tool_args)

                # Update reports and agent status based on chunk content
                # Analyst Team Reports
                if "market_report" in chunk and chunk["market_report"]:
                    log_debug_message("Agent_Status", "Market Analyst completed, updating report")
                    
                    # 更新报告和状态
                    if display.config.mode == DisplayMode.SILENT:
                        message_buffer.update_report_section("market_report", chunk["market_report"])
                        message_buffer.update_agent_status("Market Analyst", "completed")
                        console.print("[green]✓ 市场分析完成[/green]")
                    else:
                        display.update_report_section("market_report", chunk["market_report"])
                        display.update_agent_status("Market Analyst", "completed")
                        message_buffer.update_report_section("market_report", chunk["market_report"])
                        message_buffer.update_agent_status("Market Analyst", "completed")
                    
                    # Set next analyst to in_progress
                    if "social" in selections["analysts"]:
                        log_debug_message("Agent_Status", "Starting Social Analyst")
                        if display.config.mode == DisplayMode.SILENT:
                            message_buffer.update_agent_status("Social Analyst", "in_progress")
                        else:
                            display.update_agent_status("Social Analyst", "in_progress")
                            message_buffer.update_agent_status("Social Analyst", "in_progress")

                if "sentiment_report" in chunk and chunk["sentiment_report"]:
                    log_debug_message("Agent_Status", "Social Analyst completed, updating sentiment report")
                    
                    if display.config.mode == DisplayMode.SILENT:
                        message_buffer.update_report_section("sentiment_report", chunk["sentiment_report"])
                        message_buffer.update_agent_status("Social Analyst", "completed")
                        console.print("[green]✓ 情感分析完成[/green]")
                    else:
                        display.update_report_section("sentiment_report", chunk["sentiment_report"])
                        display.update_agent_status("Social Analyst", "completed")
                        message_buffer.update_report_section("sentiment_report", chunk["sentiment_report"])
                        message_buffer.update_agent_status("Social Analyst", "completed")
                    
                    # Set next analyst to in_progress
                    if "news" in selections["analysts"]:
                        log_debug_message("Agent_Status", "Starting News Analyst")
                        if display.config.mode == DisplayMode.SILENT:
                            message_buffer.update_agent_status("News Analyst", "in_progress")
                        else:
                            display.update_agent_status("News Analyst", "in_progress")
                            message_buffer.update_agent_status("News Analyst", "in_progress")

                if "news_report" in chunk and chunk["news_report"]:
                    log_debug_message("Agent_Status", "News Analyst completed, updating news report")
                    
                    if display.config.mode == DisplayMode.SILENT:
                        message_buffer.update_report_section("news_report", chunk["news_report"])
                        message_buffer.update_agent_status("News Analyst", "completed")
                        console.print("[green]✓ 新闻分析完成[/green]")
                    else:
                        display.update_report_section("news_report", chunk["news_report"])
                        display.update_agent_status("News Analyst", "completed")
                        message_buffer.update_report_section("news_report", chunk["news_report"])
                        message_buffer.update_agent_status("News Analyst", "completed")
                    
                    # Set next analyst to in_progress
                    if "fundamentals" in selections["analysts"]:
                        log_debug_message("Agent_Status", "Starting Fundamentals Analyst")
                        if display.config.mode == DisplayMode.SILENT:
                            message_buffer.update_agent_status("Fundamentals Analyst", "in_progress")
                        else:
                            display.update_agent_status("Fundamentals Analyst", "in_progress")
                            message_buffer.update_agent_status("Fundamentals Analyst", "in_progress")

                if "fundamentals_report" in chunk and chunk["fundamentals_report"]:
                    log_debug_message("Agent_Status", "Fundamentals Analyst completed, starting research team")
                    
                    if display.config.mode == DisplayMode.SILENT:
                        message_buffer.update_report_section("fundamentals_report", chunk["fundamentals_report"])
                        message_buffer.update_agent_status("Fundamentals Analyst", "completed")
                        console.print("[green]✓ 基本面分析完成[/green]")
                    else:
                        display.update_report_section("fundamentals_report", chunk["fundamentals_report"])
                        display.update_agent_status("Fundamentals Analyst", "completed")
                        message_buffer.update_report_section("fundamentals_report", chunk["fundamentals_report"])
                        message_buffer.update_agent_status("Fundamentals Analyst", "completed")
                    
                    # Set all research team members to in_progress
                    log_debug_message("Agent_Status", "Starting research team debate")
                    if display.config.mode == DisplayMode.SILENT:
                        update_research_team_status("in_progress")
                    else:
                        # 更新研究团队状态
                        for agent in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                            display.update_agent_status(agent, "in_progress")
                            message_buffer.update_agent_status(agent, "in_progress")

                # Research Team - Handle Investment Debate State
                if (
                    "investment_debate_state" in chunk
                    and chunk["investment_debate_state"]
                ):
                    debate_state = chunk["investment_debate_state"]
                    log_debug_message("Investment_Debate", f"Processing debate state with keys: {list(debate_state.keys())}")

                    # Update Bull Researcher status and report
                    if "bull_history" in debate_state and debate_state["bull_history"]:
                        log_debug_message("Bull_Researcher", "Processing bull researcher response")
                        
                        # Extract latest bull response
                        bull_responses = debate_state["bull_history"].split("\n")
                        latest_bull = bull_responses[-1] if bull_responses else ""
                        if latest_bull:
                            log_debug_message("Bull_Researcher", latest_bull)
                            
                            if display.config.mode == DisplayMode.SILENT:
                                message_buffer.add_message("Reasoning", latest_bull)
                                message_buffer.update_report_section(
                                    "investment_plan",
                                    f"### Bull Researcher Analysis\n{latest_bull}",
                                )
                                update_research_team_status("in_progress")
                            else:
                                display.log_agent("Bull Researcher", latest_bull)
                                display.update_report_section(
                                    "investment_plan",
                                    f"### Bull Researcher Analysis\n{latest_bull}",
                                )
                                message_buffer.add_message("Reasoning", latest_bull)
                                message_buffer.update_report_section(
                                    "investment_plan",
                                    f"### Bull Researcher Analysis\n{latest_bull}",
                                )
                                # 保持研究团队状态
                                for agent in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                                    display.update_agent_status(agent, "in_progress")
                                    message_buffer.update_agent_status(agent, "in_progress")

                    # Update Bear Researcher status and report
                    if "bear_history" in debate_state and debate_state["bear_history"]:
                        log_debug_message("Bear_Researcher", "Processing bear researcher response")
                        
                        # Extract latest bear response
                        bear_responses = debate_state["bear_history"].split("\n")
                        latest_bear = bear_responses[-1] if bear_responses else ""
                        if latest_bear:
                            log_debug_message("Bear_Researcher", latest_bear)
                            
                            if display.config.mode == DisplayMode.SILENT:
                                message_buffer.add_message("Reasoning", latest_bear)
                                message_buffer.update_report_section(
                                    "investment_plan",
                                    f"{message_buffer.report_sections['investment_plan']}\n\n### Bear Researcher Analysis\n{latest_bear}",
                                )
                                update_research_team_status("in_progress")
                            else:
                                display.log_agent("Bear Researcher", latest_bear)
                                display.update_report_section(
                                    "investment_plan",
                                    f"{message_buffer.report_sections['investment_plan']}\n\n### Bear Researcher Analysis\n{latest_bear}",
                                )
                                message_buffer.add_message("Reasoning", latest_bear)
                                message_buffer.update_report_section(
                                    "investment_plan",
                                    f"{message_buffer.report_sections['investment_plan']}\n\n### Bear Researcher Analysis\n{latest_bear}",
                                )
                                # 保持研究团队状态
                                for agent in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                                    display.update_agent_status(agent, "in_progress")
                                    message_buffer.update_agent_status(agent, "in_progress")

                    # Update Research Manager status and final decision
                    if (
                        "judge_decision" in debate_state
                        and debate_state["judge_decision"]
                    ):
                        log_debug_message("Research_Manager", "Processing final decision")
                        log_debug_message("Research_Manager", debate_state["judge_decision"])
                        
                        if display.config.mode == DisplayMode.SILENT:
                            message_buffer.add_message(
                                "Reasoning",
                                f"Research Manager: {debate_state['judge_decision']}",
                            )
                            message_buffer.update_report_section(
                                "investment_plan",
                                f"{message_buffer.report_sections['investment_plan']}\n\n### Research Manager Decision\n{debate_state['judge_decision']}",
                            )
                            update_research_team_status("completed")
                            message_buffer.update_agent_status("Risky Analyst", "in_progress")
                            console.print("[green]✓ 投资计划完成，开始风险管理[/green]")
                        else:
                            display.log_agent("Research Manager", f"Final Decision: {debate_state['judge_decision']}")
                            display.update_report_section(
                                "investment_plan",
                                f"{message_buffer.report_sections['investment_plan']}\n\n### Research Manager Decision\n{debate_state['judge_decision']}",
                            )
                            message_buffer.add_message(
                                "Reasoning",
                                f"Research Manager: {debate_state['judge_decision']}",
                            )
                            message_buffer.update_report_section(
                                "investment_plan",
                                f"{message_buffer.report_sections['investment_plan']}\n\n### Research Manager Decision\n{debate_state['judge_decision']}",
                            )
                            # 完成研究团队，开始风险管理
                            for agent in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                                display.update_agent_status(agent, "completed")
                                message_buffer.update_agent_status(agent, "completed")
                            display.update_agent_status("Risky Analyst", "in_progress")
                            message_buffer.update_agent_status("Risky Analyst", "in_progress")
                        
                        log_debug_message("Agent_Status", "Research team completed, starting risk management")

                # Trading Team
                if (
                    "trader_investment_plan" in chunk
                    and chunk["trader_investment_plan"]
                ):
                    message_buffer.update_report_section(
                        "trader_investment_plan", chunk["trader_investment_plan"]
                    )
                    # Set first risk analyst to in_progress
                    message_buffer.update_agent_status("Risky Analyst", "in_progress")

                # Risk Management Team - Handle Risk Debate State
                if "risk_debate_state" in chunk and chunk["risk_debate_state"]:
                    risk_state = chunk["risk_debate_state"]
                    log_debug_message("Risk_Debate", f"Processing risk debate state with keys: {list(risk_state.keys())}")

                    # Update Risky Analyst status and report
                    if (
                        "current_risky_response" in risk_state
                        and risk_state["current_risky_response"]
                    ):
                        log_debug_message("Risky_Analyst", "Processing risky analyst response")
                        message_buffer.update_agent_status(
                            "Risky Analyst", "in_progress"
                        )
                        log_debug_message("Risky_Analyst", risk_state["current_risky_response"])
                        message_buffer.add_message(
                            "Reasoning",
                            f"Risky Analyst: {risk_state['current_risky_response']}",
                        )
                        # Update risk report with risky analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Risky Analyst Analysis\n{risk_state['current_risky_response']}",
                        )

                    # Update Safe Analyst status and report
                    if (
                        "current_safe_response" in risk_state
                        and risk_state["current_safe_response"]
                    ):
                        log_debug_message("Safe_Analyst", "Processing safe analyst response")
                        message_buffer.update_agent_status(
                            "Safe Analyst", "in_progress"
                        )
                        log_debug_message("Safe_Analyst", risk_state["current_safe_response"])
                        message_buffer.add_message(
                            "Reasoning",
                            f"Safe Analyst: {risk_state['current_safe_response']}",
                        )
                        # Update risk report with safe analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Safe Analyst Analysis\n{risk_state['current_safe_response']}",
                        )

                    # Update Neutral Analyst status and report
                    if (
                        "current_neutral_response" in risk_state
                        and risk_state["current_neutral_response"]
                    ):
                        log_debug_message("Neutral_Analyst", "Processing neutral analyst response")
                        message_buffer.update_agent_status(
                            "Neutral Analyst", "in_progress"
                        )
                        log_debug_message("Neutral_Analyst", risk_state["current_neutral_response"])
                        message_buffer.add_message(
                            "Reasoning",
                            f"Neutral Analyst: {risk_state['current_neutral_response']}",
                        )
                        # Update risk report with neutral analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Neutral Analyst Analysis\n{risk_state['current_neutral_response']}",
                        )

                    # Update Portfolio Manager status and final decision
                    if "judge_decision" in risk_state and risk_state["judge_decision"]:
                        log_debug_message("Portfolio_Manager", "Processing final portfolio decision")
                        message_buffer.update_agent_status(
                            "Portfolio Manager", "in_progress"
                        )
                        log_debug_message("Portfolio_Manager", risk_state["judge_decision"])
                        message_buffer.add_message(
                            "Reasoning",
                            f"Portfolio Manager: {risk_state['judge_decision']}",
                        )
                        # Update risk report with final decision only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Portfolio Manager Decision\n{risk_state['judge_decision']}",
                        )
                        # Mark risk analysts as completed
                        log_debug_message("Agent_Status", "All risk management team completed")
                        message_buffer.update_agent_status("Risky Analyst", "completed")
                        message_buffer.update_agent_status("Safe Analyst", "completed")
                        message_buffer.update_agent_status(
                            "Neutral Analyst", "completed"
                        )
                        message_buffer.update_agent_status(
                            "Portfolio Manager", "completed"
                        )

            trace.append(chunk)

        # 分析完成处理
        final_state = trace[-1] if trace else {}
        
        # 更新所有代理状态为完成
        for agent in agent_names:
            if display.config.mode == DisplayMode.SILENT:
                message_buffer.update_agent_status(agent, "completed")
            else:
                display.update_agent_status(agent, "completed")
                message_buffer.update_agent_status(agent, "completed")

        # 添加完成消息
        completion_msg = f"分析完成: {selections['ticker']} ({selections['analysis_date']})"
        debug_msg = f"调试消息已保存到: {debug_log_file}"
        
        if display.config.mode == DisplayMode.SILENT:
            console.print(f"[green]✓ {completion_msg}[/green]")
            console.print(f"[dim]📁 {debug_msg}[/dim]")
            message_buffer.add_message("Analysis", completion_msg)
            message_buffer.add_message("System", debug_msg)
        else:
            display.log_system(completion_msg)
            display.log_system(debug_msg)
            message_buffer.add_message("Analysis", completion_msg)
            message_buffer.add_message("System", debug_msg)

        # 更新最终报告部分
        if final_state:
            for section in message_buffer.report_sections.keys():
                if section in final_state:
                    if display.config.mode == DisplayMode.SILENT:
                        message_buffer.update_report_section(section, final_state[section])
                    else:
                        display.update_report_section(section, final_state[section])
                        message_buffer.update_report_section(section, final_state[section])

        # 显示完整的最终报告
        if display.config.mode != DisplayMode.SILENT:
            display_complete_report(final_state)
        else:
            display_complete_report(final_state)
            
    except Exception as e:
        error_msg = f"分析过程中发生错误: {str(e)}"
        if display.config.mode == DisplayMode.SILENT:
            console.print(f"[red]❌ {error_msg}[/red]")
        else:
            display.log_error(error_msg)
            display.stop()
        raise
    finally:
        # 清理显示器
        if display.config.mode != DisplayMode.SILENT and hasattr(display, 'stop'):
            display.stop()


@app.command("analyze")
def analyze(
    display_mode: str = typer.Option(
        "compact",
        "--display-mode", "-d",
        help="显示模式: full(完整), compact(紧凑), minimal(最小), silent(静默)"
    ),
    max_messages: int = typer.Option(
        15,
        "--max-messages", "-m",
        help="最大显示消息数量"
    ),
    refresh_rate: float = typer.Option(
        2.0,
        "--refresh-rate", "-r",
        help="界面刷新频率(Hz)"
    ),
    show_debug: bool = typer.Option(
        False,
        "--debug",
        help="显示调试信息"
    ),
    show_tool_details: bool = typer.Option(
        False,
        "--tool-details",
        help="显示工具调用详情"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="详细输出模式"
    )
):
    """运行交易分析"""
    # 配置显示模式
    global GLOBAL_DISPLAY_CONFIG, GLOBAL_OPTIMIZED_DISPLAY
    
    # 验证显示模式
    try:
        mode = DisplayMode(display_mode.lower())
    except ValueError:
        console.print(f"[red]错误: 无效的显示模式 '{display_mode}'[/red]")
        console.print("[yellow]可用模式: full, compact, minimal, silent[/yellow]")
        raise typer.Exit(1)
    
    # 更新全局配置
    GLOBAL_DISPLAY_CONFIG.mode = mode
    GLOBAL_DISPLAY_CONFIG.max_messages = max_messages
    GLOBAL_DISPLAY_CONFIG.refresh_rate = refresh_rate
    GLOBAL_DISPLAY_CONFIG.show_debug = show_debug or verbose
    GLOBAL_DISPLAY_CONFIG.show_tool_details = show_tool_details or verbose
    
    # 如果是详细模式，启用更多选项
    if verbose:
        GLOBAL_DISPLAY_CONFIG.show_timestamps = True
        GLOBAL_DISPLAY_CONFIG.max_content_length = 300
        GLOBAL_DISPLAY_CONFIG.message_filters[MessageType.DEBUG] = True
    
    # 创建优化显示器
    GLOBAL_OPTIMIZED_DISPLAY = OptimizedDisplay(GLOBAL_DISPLAY_CONFIG)
    
    # 运行分析
    run_analysis()


if __name__ == "__main__":
    app()
