#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志管理器

提供项目级别的统一日志配置和管理功能，支持：
- 结构化日志记录
- 多级别日志分类
- 异步日志处理
- 自动日志轮转
- 敏感信息过滤
- 性能监控
"""

import os
import re
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextvars import ContextVar
import threading
from queue import Queue, Empty
import time

# 上下文变量用于追踪
request_id: ContextVar[str] = ContextVar('request_id', default='')
session_id: ContextVar[str] = ContextVar('session_id', default='')
user_id: ContextVar[str] = ContextVar('user_id', default='')


class SensitiveDataFilter(logging.Filter):
    """敏感数据过滤器"""
    
    # 敏感字段模式
    SENSITIVE_PATTERNS = [
        (re.compile(r'(api[_-]?key[s]?["\']?\s*[:=]\s*["\']?)([^"\s,}]+)', re.IGNORECASE), r'\1****'),
        (re.compile(r'(token[s]?["\']?\s*[:=]\s*["\']?)([^"\s,}]+)', re.IGNORECASE), r'\1****'),
        (re.compile(r'(password[s]?["\']?\s*[:=]\s*["\']?)([^"\s,}]+)', re.IGNORECASE), r'\1****'),
        (re.compile(r'(secret[s]?["\']?\s*[:=]\s*["\']?)([^"\s,}]+)', re.IGNORECASE), r'\1****'),
        # API密钥格式：显示前4位和后4位
        (re.compile(r'\b([a-zA-Z0-9]{4})[a-zA-Z0-9]{8,}([a-zA-Z0-9]{4})\b'), r'\1****\2'),
    ]
    
    def filter(self, record):
        """过滤敏感信息"""
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                msg = pattern.sub(replacement, msg)
            record.msg = msg
        
        # 过滤args中的敏感信息
        if hasattr(record, 'args') and record.args:
            filtered_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern, replacement in self.SENSITIVE_PATTERNS:
                        arg = pattern.sub(replacement, arg)
                filtered_args.append(arg)
            record.args = tuple(filtered_args)
        
        return True


class StructuredFormatter(logging.Formatter):
    """结构化日志格式器"""
    
    def format(self, record):
        """格式化日志记录为JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
            'filename': record.filename,
            'line_number': record.lineno,
            'function': record.funcName,
        }
        
        # 添加上下文信息
        if request_id.get():
            log_entry['request_id'] = request_id.get()
        if session_id.get():
            log_entry['session_id'] = session_id.get()
        if user_id.get():
            log_entry['user_id'] = user_id.get()
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 添加自定义字段
        for key, value in record.__dict__.items():
            if key.startswith('custom_'):
                log_entry[key[7:]] = value  # 移除'custom_'前缀
        
        return json.dumps(log_entry, ensure_ascii=False)


class AsyncLogHandler(logging.Handler):
    """异步日志处理器"""
    
    def __init__(self, target_handler, queue_size=1000):
        super().__init__()
        self.target_handler = target_handler
        self.queue = Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def emit(self, record):
        """异步发送日志记录"""
        try:
            self.queue.put_nowait(record)
        except:
            # 队列满时丢弃日志，避免阻塞
            pass
    
    def _worker(self):
        """后台工作线程"""
        while not self._stop_event.is_set():
            try:
                record = self.queue.get(timeout=1)
                self.target_handler.emit(record)
                self.queue.task_done()
            except Empty:
                continue
            except Exception:
                # 忽略日志处理错误，避免影响主程序
                pass
    
    def close(self):
        """关闭处理器"""
        self._stop_event.set()
        self.thread.join(timeout=5)
        self.target_handler.close()
        super().close()


class LoggerManager:
    """统一日志管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self._setup_logging()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'root_level': 'INFO',
            'log_dir': 'logs',
            'max_file_size': 50 * 1024 * 1024,  # 50MB
            'backup_count': 10,
            'retention_days': 30,
            'enable_async': True,
            'enable_structured': True,
            'enable_console': False,  # 禁用控制台输出以避免日志刷屏
            'categories': {
                'trading': {'level': 'WARNING', 'file': 'trading.log'},
                'dataflow': {'level': 'WARNING', 'file': 'dataflow.log'},  # 提高dataflow日志级别
                'agents': {'level': 'INFO', 'file': 'agents.log'},
                'cli': {'level': 'INFO', 'file': 'cli.log'},
                'system': {'level': 'WARNING', 'file': 'system.log'},
                'performance': {'level': 'WARNING', 'file': 'performance.log'},
            }
        }
    
    def _setup_logging(self):
        """设置日志系统"""
        # 创建日志目录
        log_dir = Path(self.config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置根日志级别
        logging.getLogger().setLevel(self.config['root_level'])
        
        # 创建各类别的日志器
        for category, category_config in self.config['categories'].items():
            self._create_category_logger(category, category_config, log_dir)
        
        # 设置默认日志器
        self._setup_default_logger(log_dir)
    
    def _create_category_logger(self, category: str, config: Dict[str, Any], log_dir: Path):
        """创建分类日志器"""
        logger = logging.getLogger(f'tradingagents.{category}')
        logger.setLevel(config['level'])
        logger.propagate = False
        
        # 文件处理器
        file_path = log_dir / config['file']
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=self.config['max_file_size'],
            backupCount=self.config['backup_count'],
            encoding='utf-8'
        )
        
        # 设置格式器
        if self.config['enable_structured']:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        file_handler.setFormatter(formatter)
        
        # 添加敏感数据过滤器
        file_handler.addFilter(SensitiveDataFilter())
        
        # 异步处理器
        if self.config['enable_async']:
            async_handler = AsyncLogHandler(file_handler)
            logger.addHandler(async_handler)
            self.handlers[f'{category}_file'] = async_handler
        else:
            logger.addHandler(file_handler)
            self.handlers[f'{category}_file'] = file_handler
        
        # 控制台处理器（仅在启用时）
        if self.config['enable_console']:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            console_handler.addFilter(SensitiveDataFilter())
            logger.addHandler(console_handler)
            self.handlers[f'{category}_console'] = console_handler
        
        self.loggers[category] = logger
    
    def _setup_default_logger(self, log_dir: Path):
        """设置默认日志器"""
        # 为未分类的日志设置默认处理器
        root_logger = logging.getLogger()
        
        # 默认文件处理器
        default_file = log_dir / 'default.log'
        default_handler = logging.handlers.RotatingFileHandler(
            default_file,
            maxBytes=self.config['max_file_size'],
            backupCount=self.config['backup_count'],
            encoding='utf-8'
        )
        
        if self.config['enable_structured']:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        default_handler.setFormatter(formatter)
        default_handler.addFilter(SensitiveDataFilter())
        
        if self.config['enable_async']:
            async_handler = AsyncLogHandler(default_handler)
            root_logger.addHandler(async_handler)
            self.handlers['default'] = async_handler
        else:
            root_logger.addHandler(default_handler)
            self.handlers['default'] = default_handler
    
    def get_logger(self, category: str = None, name: str = None) -> logging.Logger:
        """获取日志器"""
        if category and category in self.loggers:
            if name:
                return logging.getLogger(f'tradingagents.{category}.{name}')
            return self.loggers[category]
        
        if name:
            return logging.getLogger(name)
        
        return logging.getLogger()
    
    def set_context(self, **kwargs):
        """设置日志上下文"""
        if 'request_id' in kwargs:
            request_id.set(kwargs['request_id'])
        if 'session_id' in kwargs:
            session_id.set(kwargs['session_id'])
        if 'user_id' in kwargs:
            user_id.set(kwargs['user_id'])
    
    def clear_context(self):
        """清除日志上下文"""
        request_id.set('')
        session_id.set('')
        user_id.set('')
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """记录性能日志"""
        perf_logger = self.get_logger('performance')
        perf_logger.info(
            f"Performance: {operation}",
            extra={
                'custom_operation': operation,
                'custom_duration': duration,
                'custom_timestamp': time.time(),
                **{f'custom_{k}': v for k, v in kwargs.items()}
            }
        )
    
    def cleanup_old_logs(self):
        """清理旧日志文件"""
        log_dir = Path(self.config['log_dir'])
        retention_days = self.config['retention_days']
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        
        for log_file in log_dir.glob('*.log*'):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    # 使用标准输出而不是日志记录，避免循环依赖
                except Exception as e:
                    # 使用标准输出而不是日志记录，避免循环依赖
                    pass
    
    def close(self):
        """关闭日志管理器"""
        for handler in self.handlers.values():
            handler.close()
        self.handlers.clear()
        self.loggers.clear()


# 全局日志管理器实例
_logger_manager: Optional[LoggerManager] = None


def get_logger_manager(config: Dict[str, Any] = None) -> LoggerManager:
    """获取全局日志管理器实例"""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = LoggerManager(config)
    return _logger_manager


def get_logger(category: str = None, name: str = None) -> logging.Logger:
    """便捷函数：获取日志器"""
    return get_logger_manager().get_logger(category, name)


def set_log_context(**kwargs):
    """便捷函数：设置日志上下文"""
    get_logger_manager().set_context(**kwargs)


def clear_log_context():
    """便捷函数：清除日志上下文"""
    get_logger_manager().clear_context()


def log_performance(operation: str, duration: float, **kwargs):
    """便捷函数：记录性能日志"""
    get_logger_manager().log_performance(operation, duration, **kwargs)