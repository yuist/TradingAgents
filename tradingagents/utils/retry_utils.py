#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重试和错误处理工具模块

该模块提供统一的重试机制和错误处理功能，用于提高系统的稳定性和容错能力。
"""

import time
from typing import Any, Callable, Optional, Type, Union, Tuple
from functools import wraps
from threading import Lock
from collections import defaultdict
from openai import APITimeoutError, APIError, RateLimitError
from requests.exceptions import RequestException, Timeout, ConnectionError

# 尝试导入yfinance的异常类型
try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    # 如果yfinance没有这个异常类型，创建一个占位符
    class YFRateLimitError(Exception):
        pass

from .logging_manager import get_logger

# 创建日志器
logger = get_logger('utils', 'retry')

# 频率控制：记录每个API的最后调用时间
_api_call_tracker = defaultdict(float)
_api_call_lock = Lock()

# 定义可重试的异常类型
RETRIABLE_EXCEPTIONS = (
    APITimeoutError,
    RateLimitError,
    YFRateLimitError,  # 添加Yahoo Finance频率限制异常
    Timeout,
    ConnectionError,
    RequestException,
)

# 定义不可重试的异常类型
NON_RETRIABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
)


def rate_limit_control(api_name: str, min_interval: float = 12.0):
    """
    API频率控制函数
    
    Args:
        api_name: API名称，用于跟踪不同API的调用频率
        min_interval: 最小调用间隔（秒），默认12秒（每分钟5次）
    """
    with _api_call_lock:
        current_time = time.time()
        last_call_time = _api_call_tracker[api_name]
        
        if last_call_time > 0:  # 不是第一次调用
            time_since_last_call = current_time - last_call_time
            if time_since_last_call < min_interval:
                sleep_time = min_interval - time_since_last_call
                logger.info(f"API频率控制: {api_name} 需要等待 {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        _api_call_tracker[api_name] = time.time()


def with_yahoo_finance_retry(
    max_retries: int = 2,  # 减少重试次数
    initial_delay: float = 5.0,  # 减少到5秒初始延迟
    max_delay: float = 30.0,     # 减少到30秒最大延迟
    backoff_factor: float = 2.0,
    rate_limit_interval: float = 3.0,  # 减少到3秒间隔
    log_errors: bool = True
):
    """
    专门针对Yahoo Finance API的重试装饰器
    
    Args:
        max_retries: 最大重试次数，默认5次
        initial_delay: 初始延迟时间（秒），默认10秒
        max_delay: 最大延迟时间（秒），默认300秒（5分钟）
        backoff_factor: 退避因子，默认2.0
        rate_limit_interval: API调用间隔（秒），默认12秒
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 应用频率控制
            rate_limit_control(f"yahoo_finance_{func.__name__}", rate_limit_interval)
            
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(f"Yahoo Finance API {func.__name__} 在第 {attempt + 1} 次尝试后成功")
                    
                    return result
                    
                except (YFRateLimitError, Exception) as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    # 检查是否是频率限制错误
                    is_rate_limit = (
                        isinstance(e, YFRateLimitError) or
                        'rate limit' in error_msg or
                        'too many requests' in error_msg or
                        '429' in error_msg
                    )
                    
                    if attempt < max_retries:
                        if is_rate_limit:
                            # 频率限制错误：使用更长的延迟
                            delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                            logger.warning(
                                f"Yahoo Finance API频率限制 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}. "
                                f"将等待 {delay:.1f}s 后重试..."
                            )
                        elif 'timeout' in error_msg or 'connection' in error_msg:
                            # 网络错误：使用较短的延迟
                            delay = min(5.0 * (1.5 ** attempt), 60.0)
                            logger.warning(
                                f"Yahoo Finance API网络错误 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}. "
                                f"将等待 {delay:.1f}s 后重试..."
                            )
                        else:
                            # 其他错误：使用标准延迟
                            delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                            logger.warning(
                                f"Yahoo Finance API错误 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}. "
                                f"将等待 {delay:.1f}s 后重试..."
                            )
                        
                        time.sleep(delay)
                    else:
                        logger.error(f"Yahoo Finance API {func.__name__} 达到最大重试次数 ({max_retries})")
                        break
            
            # 所有重试都失败了，抛出最后一个异常
            if last_exception:
                if isinstance(last_exception, YFRateLimitError) or 'rate limit' in str(last_exception).lower():
                    raise Exception(
                        f"Yahoo Finance API频率限制：已达到最大重试次数。请等待更长时间后再试，或考虑使用其他数据源。原始错误: {last_exception}"
                    )
                else:
                    raise last_exception
            else:
                raise RuntimeError(f"Yahoo Finance API {func.__name__} 执行失败，原因未知")
        
        return wrapper
    return decorator


def with_retry(
    max_retries: int = 3,
    delay: float = 2.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = RETRIABLE_EXCEPTIONS,
    non_retriable: Tuple[Type[Exception], ...] = NON_RETRIABLE_EXCEPTIONS,
    timeout: Optional[float] = None,
    fallback_value: Any = None,
    log_errors: bool = True
):
    """
    重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff_factor: 退避因子，每次重试延迟时间乘以此因子
        exceptions: 可重试的异常类型
        non_retriable: 不可重试的异常类型，遇到这些异常直接抛出
        timeout: 总超时时间（秒）
        fallback_value: 失败时的回退值
        log_errors: 是否记录错误日志
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    # 检查总超时
                    if timeout and (time.time() - start_time) > timeout:
                        if log_errors:
                            logger.error(f"函数 {func.__name__} 总超时 ({timeout}s)")
                        break
                    
                    # 执行函数
                    result = func(*args, **kwargs)
                    
                    # 成功执行，记录重试信息
                    if attempt > 0 and log_errors:
                        logger.info(f"函数 {func.__name__} 在第 {attempt + 1} 次尝试后成功")
                    
                    return result
                    
                except non_retriable as e:
                    # 不可重试的异常，直接抛出
                    if log_errors:
                        logger.error(f"函数 {func.__name__} 遇到不可重试异常: {type(e).__name__}: {e}")
                    raise e
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        # 计算延迟时间
                        current_delay = delay * (backoff_factor ** attempt)
                        
                        if log_errors:
                            logger.warning(
                                f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {type(e).__name__}: {e}. "
                                f"将在 {current_delay:.1f}s 后重试..."
                            )
                        
                        # 检查是否还有时间重试
                        if timeout and (time.time() - start_time + current_delay) > timeout:
                            if log_errors:
                                logger.error(f"函数 {func.__name__} 重试会超过总超时时间，停止重试")
                            break
                        
                        time.sleep(current_delay)
                    else:
                        if log_errors:
                            logger.error(f"函数 {func.__name__} 达到最大重试次数 ({max_retries})")
                        break
                        
                except Exception as e:
                    # 其他未预期的异常
                    if log_errors:
                        logger.error(f"函数 {func.__name__} 遇到未预期异常: {type(e).__name__}: {e}")
                    raise e
            
            # 所有重试都失败了
            if fallback_value is not None:
                if log_errors:
                    logger.warning(f"函数 {func.__name__} 所有重试失败，返回回退值: {fallback_value}")
                return fallback_value
            else:
                # 抛出最后一个异常
                if last_exception:
                    raise last_exception
                else:
                    raise RuntimeError(f"函数 {func.__name__} 执行失败，原因未知")
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    max_retries: int = 3,
    delay: float = 2.0,
    fallback_value: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    安全执行函数，带重试机制
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        max_retries: 最大重试次数
        delay: 重试延迟时间
        fallback_value: 失败时的回退值
        log_errors: 是否记录错误日志
        **kwargs: 函数关键字参数
    
    Returns:
        函数执行结果或回退值
    """
    @with_retry(
        max_retries=max_retries,
        delay=delay,
        fallback_value=fallback_value,
        log_errors=log_errors
    )
    def _execute():
        return func(*args, **kwargs)
    
    return _execute()


def get_error_message(exception: Exception) -> str:
    """
    获取友好的错误消息
    
    Args:
        exception: 异常对象
    
    Returns:
        str: 友好的错误消息
    """
    error_type = type(exception).__name__
    error_msg = str(exception)
    
    # 针对不同类型的异常提供更友好的消息
    if isinstance(exception, APITimeoutError):
        return f"API调用超时: {error_msg}。建议检查网络连接或增加超时时间。"
    elif isinstance(exception, RateLimitError):
        return f"API调用频率限制: {error_msg}。建议稍后重试或检查API配额。"
    elif isinstance(exception, ConnectionError):
        return f"网络连接错误: {error_msg}。建议检查网络连接。"
    elif isinstance(exception, Timeout):
        return f"请求超时: {error_msg}。建议检查网络状况或增加超时时间。"
    elif isinstance(exception, KeyError):
        return f"配置错误: 缺少必要的配置项 {error_msg}。请检查配置文件。"
    elif isinstance(exception, ValueError):
        return f"参数错误: {error_msg}。请检查输入参数。"
    else:
        return f"{error_type}: {error_msg}"


def log_performance(func: Callable) -> Callable:
    """
    性能监控装饰器
    
    Args:
        func: 要监控的函数
    
    Returns:
        装饰后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"函数 {func.__name__} 执行成功，耗时: {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"函数 {func.__name__} 执行失败，耗时: {execution_time:.2f}s，错误: {get_error_message(e)}")
            raise
    return wrapper