#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI优化显示模块
解决刷屏问题，提供最佳实践的动态信息展示

主要优化:
1. 智能消息过滤和缓冲
2. 分级日志输出控制
3. 动态刷新率调整
4. 内容长度限制
5. 静默模式支持
"""

import os
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.spinner import Spinner
from rich.markdown import Markdown
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import datetime


class MessageType(Enum):
    """消息类型枚举"""
    SYSTEM = "System"
    REASONING = "Reasoning"
    TOOL = "Tool"
    AGENT_STATUS = "Agent"
    REPORT = "Report"
    ERROR = "Error"
    DEBUG = "Debug"
    AGENT_MESSAGE = "Agent_Message"  # 新增：专门用于agent消息
    SYSTEM_LOG = "System_Log"        # 新增：专门用于系统日志


class DisplayMode(Enum):
    """显示模式枚举"""
    FULL = "full"          # 完整显示模式
    COMPACT = "compact"    # 紧凑显示模式
    MINIMAL = "minimal"    # 最小显示模式
    SILENT = "silent"      # 静默模式


@dataclass
class DisplayConfig:
    """显示配置"""
    mode: DisplayMode = DisplayMode.COMPACT
    max_messages: int = 15
    max_content_length: int = 150
    refresh_rate: float = 2.0
    show_timestamps: bool = True
    show_debug: bool = False
    show_tool_details: bool = False
    enable_progress_bar: bool = True
    auto_scroll: bool = True
    filter_duplicates: bool = True
    
    # 消息类型过滤
    message_filters: Dict[MessageType, bool] = field(default_factory=lambda: {
        MessageType.SYSTEM: True,
        MessageType.REASONING: True,
        MessageType.TOOL: True,
        MessageType.AGENT_STATUS: True,
        MessageType.REPORT: True,
        MessageType.ERROR: True,
        MessageType.DEBUG: False,
    })


@dataclass
class Message:
    """消息数据结构"""
    timestamp: str
    msg_type: MessageType
    content: str
    agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # 内容长度限制
        if len(self.content) > 500:
            self.content = self.content[:497] + "..."


class OptimizedMessageBuffer:
    """优化的消息缓冲区"""
    
    def __init__(self, config: DisplayConfig):
        self.config = config
        self.messages: deque = deque(maxlen=config.max_messages * 2)  # 保留更多历史
        self.agent_messages: deque = deque(maxlen=config.max_messages)  # 专门存储agent消息
        self.system_logs: deque = deque(maxlen=config.max_messages)     # 专门存储系统日志
        self.tool_calls: deque = deque(maxlen=50)
        self.agent_status: Dict[str, str] = {}
        self.report_sections: Dict[str, str] = {}
        self.current_agent: Optional[str] = None
        self.stats = {
            'total_messages': 0,
            'agent_messages': 0,
            'system_logs': 0,
            'tool_calls': 0,
            'agent_updates': 0,
            'reports_generated': 0
        }
        self._lock = threading.Lock()
        self._last_message_hash = None
        
    def add_message(self, msg_type: MessageType, content: str, agent: str = None, **metadata):
        """添加消息（线程安全）"""
        with self._lock:
            # 去重检查
            if self.config.filter_duplicates:
                content_hash = hash(content)
                if content_hash == self._last_message_hash:
                    return
                self._last_message_hash = content_hash
            
            # 过滤检查
            if not self.config.message_filters.get(msg_type, True):
                return
                
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            message = Message(
                timestamp=timestamp,
                msg_type=msg_type,
                content=content,
                agent=agent,
                metadata=metadata
            )
            
            # 根据消息类型分别存储
            if msg_type in [MessageType.REASONING, MessageType.AGENT_MESSAGE, MessageType.AGENT_STATUS]:
                self.agent_messages.append(message)
                self.stats['agent_messages'] += 1
            elif msg_type in [MessageType.SYSTEM, MessageType.SYSTEM_LOG, MessageType.TOOL, MessageType.ERROR, MessageType.DEBUG]:
                self.system_logs.append(message)
                self.stats['system_logs'] += 1
            
            # 保持原有的统一存储以兼容现有代码
            self.messages.append(message)
            self.stats['total_messages'] += 1
            
    def add_tool_call(self, tool_name: str, args: Any, agent: str = None):
        """添加工具调用"""
        with self._lock:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.tool_calls.append((timestamp, tool_name, args, agent))
            self.stats['tool_calls'] += 1
            
    def update_agent_status(self, agent: str, status: str):
        """更新代理状态"""
        with self._lock:
            self.agent_status[agent] = status
            self.current_agent = agent
            self.stats['agent_updates'] += 1
            
    def update_report_section(self, section: str, content: str):
        """更新报告部分"""
        with self._lock:
            self.report_sections[section] = content
            self.stats['reports_generated'] += 1
            
    def get_filtered_messages(self) -> List[Message]:
        """获取过滤后的消息"""
        with self._lock:
            messages = list(self.messages)
            
        # 根据显示模式过滤
        if self.config.mode == DisplayMode.MINIMAL:
            # 只显示系统消息和错误
            messages = [m for m in messages if m.msg_type in [MessageType.SYSTEM, MessageType.ERROR, MessageType.AGENT_STATUS]]
        elif self.config.mode == DisplayMode.COMPACT:
            # 排除调试消息
            messages = [m for m in messages if m.msg_type != MessageType.DEBUG]
            
        return messages[-self.config.max_messages:]


class OptimizedDisplay:
    """优化的CLI显示器"""
    
    def __init__(self, config: DisplayConfig = None):
        self.config = config or DisplayConfig()
        self.console = Console()
        self.message_buffer = OptimizedMessageBuffer(self.config)
        self.layout = self._create_layout()
        self.live = None
        self._running = False
        self._update_thread = None
        
        # 进度跟踪
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=True
        )
        
    def _create_layout(self) -> Layout:
        """创建布局"""
        layout = Layout()
        
        if self.config.mode == DisplayMode.MINIMAL:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="status", size=8),
                Layout(name="footer", size=2)
            )
        elif self.config.mode == DisplayMode.COMPACT:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=3)
            )
            layout["main"].split_row(
                Layout(name="status", ratio=1),
                Layout(name="right_panel", ratio=2)
            )
            layout["right_panel"].split_column(
                Layout(name="agent_messages", ratio=1),
                Layout(name="system_logs", ratio=1)
            )
        else:  # FULL mode
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=3)
            )
            layout["main"].split_row(
                Layout(name="left", ratio=2),
                Layout(name="right", ratio=1)
            )
            layout["left"].split_column(
                Layout(name="status", size=8),
                Layout(name="messages", ratio=1)
            )
            layout["right"].split_column(
                Layout(name="analysis", ratio=1)
            )
            
        return layout
        
    def _update_header(self):
        """更新头部"""
        if self.config.mode == DisplayMode.SILENT:
            return
            
        header_content = "[bold green]TradingAgents CLI[/bold green]"
        if self.config.mode != DisplayMode.MINIMAL:
            header_content += "\n[dim]智能多代理金融交易框架[/dim]"
            
        self.layout["header"].update(
            Panel(
                header_content,
                title="TradingAgents",
                border_style="green",
                padding=(0, 2)
            )
        )
        
    def _update_status(self):
        """更新状态面板"""
        # 安全检查布局组件是否存在
        try:
            status_layout = self.layout["status"]
        except KeyError:
            return
            
        status_table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.SIMPLE_HEAD,
            padding=(0, 1),
            expand=True
        )
        
        if self.config.mode == DisplayMode.MINIMAL:
            status_table.add_column("代理", style="cyan", width=20)
            status_table.add_column("状态", style="green", width=15)
        else:
            status_table.add_column("团队", style="cyan", width=15)
            status_table.add_column("代理", style="green", width=20)
            status_table.add_column("状态", style="yellow", width=15)
            
        # 代理团队分组
        teams = {
            "分析团队": ["Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst"],
            "研究团队": ["Bull Researcher", "Bear Researcher", "Research Manager"],
            "交易团队": ["Trader"],
            "风险管理": ["Risky Analyst", "Neutral Analyst", "Safe Analyst"],
            "投资组合": ["Portfolio Manager"]
        }
        
        for team, agents in teams.items():
            for i, agent in enumerate(agents):
                status = self.message_buffer.agent_status.get(agent, "pending")
                
                # 状态显示
                if status == "in_progress":
                    if self.config.enable_progress_bar:
                        status_display = Spinner("dots", text="[blue]进行中[/blue]")
                    else:
                        status_display = "[blue]进行中[/blue]"
                else:
                    status_colors = {
                        "pending": "yellow",
                        "completed": "green",
                        "error": "red"
                    }
                    color = status_colors.get(status, "white")
                    status_display = f"[{color}]{status}[/{color}]"
                
                if self.config.mode == DisplayMode.MINIMAL:
                    status_table.add_row(agent, status_display)
                else:
                    team_name = team if i == 0 else ""
                    status_table.add_row(team_name, agent, status_display)
                    
        self.layout["status"].update(
            Panel(
                status_table,
                title="执行状态",
                border_style="cyan",
                padding=(1, 1)
            )
        )
        
    def _update_agent_messages(self):
        """更新Agent消息面板"""
        # 安全检查布局组件是否存在
        try:
            agent_messages_layout = self.layout["agent_messages"]
        except KeyError:
            return
            
        if self.config.mode == DisplayMode.MINIMAL:
            return
            
        messages_table = Table(
            show_header=True,
            header_style="bold green",
            box=box.MINIMAL,
            show_lines=False,
            padding=(0, 1),
            expand=True
        )
        
        if self.config.show_timestamps:
            messages_table.add_column("时间", style="cyan", width=8)
        messages_table.add_column("代理", style="green", width=12)
        messages_table.add_column("内容", style="white", ratio=1)
        
        # 获取agent消息
        agent_messages = list(self.message_buffer.agent_messages)
        
        for message in agent_messages[-self.config.max_messages:]:
            # 内容截断
            content = message.content
            if len(content) > self.config.max_content_length:
                content = content[:self.config.max_content_length-3] + "..."
                
            # 格式化内容
            wrapped_content = Text(content, overflow="fold")
            agent_name = message.agent or "Unknown"
            
            row_data = []
            if self.config.show_timestamps:
                row_data.append(message.timestamp)
            row_data.extend([agent_name, wrapped_content])
            
            messages_table.add_row(*row_data)
            
        # 添加统计信息
        total = self.message_buffer.stats['agent_messages']
        shown = len(agent_messages[-self.config.max_messages:])
        messages_table.caption = f"[dim]显示最近 {shown}/{total} 条Agent消息[/dim]"
            
        self.layout["agent_messages"].update(
            Panel(
                messages_table,
                title="Agent消息",
                border_style="green",
                padding=(1, 1)
            )
        )
        
    def _update_system_logs(self):
        """更新系统日志面板"""
        # 安全检查布局组件是否存在
        try:
            system_logs_layout = self.layout["system_logs"]
        except KeyError:
            return
            
        if self.config.mode == DisplayMode.MINIMAL:
            return
            
        logs_table = Table(
            show_header=True,
            header_style="bold blue",
            box=box.MINIMAL,
            show_lines=False,
            padding=(0, 1),
            expand=True
        )
        
        if self.config.show_timestamps:
            logs_table.add_column("时间", style="cyan", width=8)
        logs_table.add_column("类型", style="blue", width=10)
        logs_table.add_column("内容", style="white", ratio=1)
        
        # 获取系统日志
        system_logs = list(self.message_buffer.system_logs)
        
        for message in system_logs[-self.config.max_messages:]:
            # 内容截断
            content = message.content
            if len(content) > self.config.max_content_length:
                content = content[:self.config.max_content_length-3] + "..."
                
            # 格式化内容
            wrapped_content = Text(content, overflow="fold")
            
            row_data = []
            if self.config.show_timestamps:
                row_data.append(message.timestamp)
            row_data.extend([message.msg_type.value, wrapped_content])
            
            logs_table.add_row(*row_data)
            
        # 添加统计信息
        total = self.message_buffer.stats['system_logs']
        shown = len(system_logs[-self.config.max_messages:])
        logs_table.caption = f"[dim]显示最近 {shown}/{total} 条系统日志[/dim]"
            
        self.layout["system_logs"].update(
            Panel(
                logs_table,
                title="系统日志",
                border_style="blue",
                padding=(1, 1)
            )
        )
        
    def _update_messages(self):
        """更新消息面板（兼容性方法）"""
        # 为了保持向后兼容性，同时更新两个面板
        self._update_agent_messages()
        self._update_system_logs()
        
    def _update_analysis(self):
        """更新分析面板"""
        # 安全检查布局组件是否存在
        try:
            analysis_layout = self.layout["analysis"]
        except KeyError:
            return
            
        # 获取最新报告
        latest_report = None
        for section, content in self.message_buffer.report_sections.items():
            if content:
                latest_report = content
                break
                
        if latest_report:
            # 限制报告长度
            if len(latest_report) > 2000:
                latest_report = latest_report[:1997] + "..."
                
            self.layout["analysis"].update(
                Panel(
                    Markdown(latest_report),
                    title="当前分析报告",
                    border_style="green",
                    padding=(1, 1)
                )
            )
        else:
            self.layout["analysis"].update(
                Panel(
                    "[italic]等待分析报告...[/italic]",
                    title="当前分析报告",
                    border_style="green",
                    padding=(1, 1)
                )
            )
            
    def _update_footer(self):
        """更新底部统计"""
        # 安全检查布局组件是否存在
        try:
            footer_layout = self.layout["footer"]
        except KeyError:
            return
            
        stats = self.message_buffer.stats
        stats_text = (
            f"消息: {stats['total_messages']} | "
            f"工具调用: {stats['tool_calls']} | "
            f"代理更新: {stats['agent_updates']} | "
            f"报告: {stats['reports_generated']}"
        )
        
        if self.config.mode != DisplayMode.MINIMAL:
            stats_text += f" | 模式: {self.config.mode.value}"
            
        self.layout["footer"].update(
            Panel(
                Text(stats_text, justify="center"),
                border_style="grey50"
            )
        )
        
    def _update_display(self):
        """更新整个显示"""
        if self.config.mode == DisplayMode.SILENT:
            return
            
        try:
            self._update_header()
            self._update_status()
            
            if self.config.mode == DisplayMode.COMPACT:
                # COMPACT模式下分别更新agent消息和系统日志
                self._update_agent_messages()
                self._update_system_logs()
            elif self.config.mode == DisplayMode.FULL:
                # FULL模式下使用原有的消息面板
                self._update_messages()
                self._update_analysis()
            elif self.config.mode == DisplayMode.MINIMAL:
                # MINIMAL模式只更新基本信息
                pass
                
            self._update_footer()
        except Exception as e:
            # 静默处理显示错误，避免影响主流程
            pass
            
    def start(self):
        """启动显示"""
        if self.config.mode == DisplayMode.SILENT:
            return
            
        self._running = True
        self.live = Live(
            self.layout,
            refresh_per_second=self.config.refresh_rate,
            console=self.console
        )
        self.live.start()
        
        # 启动更新线程
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        
    def stop(self):
        """停止显示"""
        self._running = False
        if self.live:
            self.live.stop()
            
    def _update_loop(self):
        """更新循环"""
        while self._running:
            self._update_display()
            time.sleep(1.0 / self.config.refresh_rate)
            
    # 便捷方法
    def log_system(self, message: str, agent: str = None):
        """记录系统消息"""
        self.message_buffer.add_message(MessageType.SYSTEM, message, agent)
        
    def log_reasoning(self, message: str, agent: str = None):
        """记录推理消息"""
        self.message_buffer.add_message(MessageType.REASONING, message, agent)
        
    def log_tool_call(self, tool_name: str, args: Any, agent: str = None):
        """记录工具调用"""
        self.message_buffer.add_tool_call(tool_name, args, agent)
        
    def log_error(self, message: str, agent: str = None):
        """记录错误消息"""
        self.message_buffer.add_message(MessageType.ERROR, message, agent)
        
    def update_agent_status(self, agent: str, status: str):
        """更新代理状态"""
        self.message_buffer.update_agent_status(agent, status)
        
    def update_report(self, section: str, content: str):
        """更新报告"""
        self.message_buffer.update_report_section(section, content)
        
    def update_report_section(self, section: str, content: str):
        """更新报告部分（别名方法）"""
        self.message_buffer.update_report_section(section, content)
        
    def log_agent(self, agent: str, message: str):
        """记录代理消息"""
        self.message_buffer.add_message(MessageType.REASONING, message, agent)
        
    def log_tool(self, message: str, agent: str = None):
        """记录工具消息"""
        self.message_buffer.add_message(MessageType.TOOL, message, agent)


def create_display_from_env() -> OptimizedDisplay:
    """从环境变量创建显示器"""
    config = DisplayConfig()
    
    # 从环境变量读取配置
    mode = os.getenv('CLI_DISPLAY_MODE', 'compact').lower()
    if mode in [m.value for m in DisplayMode]:
        config.mode = DisplayMode(mode)
        
    config.max_messages = int(os.getenv('CLI_MAX_MESSAGES', '15'))
    config.max_content_length = int(os.getenv('CLI_MAX_CONTENT_LENGTH', '150'))
    config.refresh_rate = float(os.getenv('CLI_REFRESH_RATE', '2.0'))
    config.show_debug = os.getenv('CLI_SHOW_DEBUG', 'false').lower() == 'true'
    config.show_tool_details = os.getenv('CLI_SHOW_TOOL_DETAILS', 'false').lower() == 'true'
    
    return OptimizedDisplay(config)


# 全局显示器实例
_global_display: Optional[OptimizedDisplay] = None


def get_display() -> OptimizedDisplay:
    """获取全局显示器实例"""
    global _global_display
    if _global_display is None:
        _global_display = create_display_from_env()
    return _global_display


def set_display_mode(mode: DisplayMode):
    """设置显示模式"""
    display = get_display()
    display.config.mode = mode
    display.layout = display._create_layout()