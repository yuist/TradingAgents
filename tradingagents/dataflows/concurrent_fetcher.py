#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发数据获取器
实现并行请求多个数据源，提高数据获取效率
"""

import asyncio
import concurrent.futures
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from ..utils.logging_manager import get_logger

logger = get_logger('dataflow', 'concurrent_fetcher')

@dataclass
class FetchResult:
    """获取结果"""
    success: bool
    data: Optional[pd.DataFrame]
    source: str
    response_time: float
    error: Optional[str] = None
    
    def is_valid(self) -> bool:
        """检查结果是否有效"""
        return (self.success and 
                self.data is not None and 
                hasattr(self.data, 'empty') and 
                not self.data.empty)

class ConcurrentDataFetcher:
    """并发数据获取器"""
    
    def __init__(self, max_workers: int = 3, timeout: int = 10):
        self.max_workers = max_workers
        self.timeout = timeout
    
    def fetch_data_parallel(self, 
                          providers: List[Any], 
                          symbol: str, 
                          start_date: str, 
                          end_date: str,
                          validation_func: Optional[Callable] = None) -> Optional[FetchResult]:
        """并行获取数据"""
        if not providers:
            return None
        
        logger.info(f"开始并行获取 {symbol} 数据，使用 {len(providers)} 个数据源")
        
        # 使用线程池并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_provider = {}
            for provider in providers:
                future = executor.submit(
                    self._fetch_single_source, 
                    provider, symbol, start_date, end_date, validation_func
                )
                future_to_provider[future] = provider
            
            # 等待第一个成功的结果
            try:
                for future in concurrent.futures.as_completed(future_to_provider, timeout=self.timeout):
                    provider = future_to_provider[future]
                    try:
                        result = future.result()
                        if result.is_valid():
                            logger.info(f"并行获取成功: {result.source}, 响应时间: {result.response_time:.2f}秒")
                            # 取消其他未完成的任务
                            for f in future_to_provider:
                                if not f.done():
                                    f.cancel()
                            return result
                        else:
                            logger.warning(f"数据源 {result.source} 返回无效数据: {result.error}")
                    except Exception as e:
                        logger.warning(f"数据源 {provider.name} 执行异常: {e}")
                        continue
            
            except concurrent.futures.TimeoutError:
                logger.error(f"并行获取超时 ({self.timeout}秒)")
                # 取消所有未完成的任务
                for future in future_to_provider:
                    future.cancel()
        
        logger.error(f"所有数据源并行获取均失败: {symbol}")
        return None
    
    def _fetch_single_source(self, 
                           provider: Any, 
                           symbol: str, 
                           start_date: str, 
                           end_date: str,
                           validation_func: Optional[Callable] = None) -> FetchResult:
        """从单个数据源获取数据"""
        start_time = time.time()
        
        try:
            logger.debug(f"开始从 {provider.name} 获取数据")
            data = provider.get_stock_data(symbol, start_date, end_date)
            response_time = time.time() - start_time
            
            # 数据验证
            if validation_func and not validation_func(data, symbol):
                return FetchResult(
                    success=False,
                    data=None,
                    source=provider.name,
                    response_time=response_time,
                    error="数据验证失败"
                )
            
            return FetchResult(
                success=True,
                data=data,
                source=provider.name,
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.debug(f"数据源 {provider.name} 获取失败: {e}")
            return FetchResult(
                success=False,
                data=None,
                source=provider.name,
                response_time=response_time,
                error=str(e)
            )
    
    def fetch_data_sequential_with_timeout(self, 
                                         providers: List[Any], 
                                         symbol: str, 
                                         start_date: str, 
                                         end_date: str,
                                         validation_func: Optional[Callable] = None,
                                         per_source_timeout: int = 5) -> Optional[FetchResult]:
        """串行获取数据但每个数据源有独立超时"""
        if not providers:
            return None
        
        logger.info(f"开始串行获取 {symbol} 数据，每个数据源超时 {per_source_timeout} 秒")
        
        for provider in providers:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._fetch_single_source, 
                    provider, symbol, start_date, end_date, validation_func
                )
                
                try:
                    result = future.result(timeout=per_source_timeout)
                    if result.is_valid():
                        logger.info(f"串行获取成功: {result.source}, 响应时间: {result.response_time:.2f}秒")
                        return result
                    else:
                        logger.warning(f"数据源 {result.source} 返回无效数据: {result.error}")
                        continue
                        
                except concurrent.futures.TimeoutError:
                    logger.warning(f"数据源 {provider.name} 超时 ({per_source_timeout}秒)")
                    future.cancel()
                    continue
                except Exception as e:
                    logger.warning(f"数据源 {provider.name} 异常: {e}")
                    continue
        
        logger.error(f"所有数据源串行获取均失败: {symbol}")
        return None
    
    def fetch_data_hybrid(self, 
                         high_priority_providers: List[Any],
                         low_priority_providers: List[Any],
                         symbol: str, 
                         start_date: str, 
                         end_date: str,
                         validation_func: Optional[Callable] = None) -> Optional[FetchResult]:
        """混合获取策略：高优先级并行，低优先级串行"""
        logger.info(f"开始混合获取 {symbol} 数据")
        
        # 首先并行尝试高优先级数据源
        if high_priority_providers:
            logger.info(f"并行尝试 {len(high_priority_providers)} 个高优先级数据源")
            result = self.fetch_data_parallel(
                high_priority_providers, symbol, start_date, end_date, validation_func
            )
            if result and result.is_valid():
                return result
        
        # 如果高优先级失败，串行尝试低优先级数据源
        if low_priority_providers:
            logger.info(f"串行尝试 {len(low_priority_providers)} 个低优先级数据源")
            result = self.fetch_data_sequential_with_timeout(
                low_priority_providers, symbol, start_date, end_date, validation_func, per_source_timeout=8
            )
            if result and result.is_valid():
                return result
        
        logger.error(f"混合获取策略失败: {symbol}")
        return None
    
    def fetch_realtime_data_hybrid(self, 
                                 high_priority_providers: List[Any],
                                 low_priority_providers: List[Any],
                                 symbol: str,
                                 timeout: int = 5) -> Optional[FetchResult]:
        """混合获取实时数据：高优先级并行，低优先级串行"""
        logger.info(f"开始混合获取 {symbol} 实时数据")
        
        # 首先并行尝试高优先级数据源
        if high_priority_providers:
            logger.info(f"并行尝试 {len(high_priority_providers)} 个高优先级实时数据源")
            result = self.fetch_realtime_data_parallel(
                high_priority_providers, symbol, timeout
            )
            if result and result.is_valid():
                return result
        
        # 如果高优先级失败，串行尝试低优先级数据源
        if low_priority_providers:
            logger.info(f"串行尝试 {len(low_priority_providers)} 个低优先级实时数据源")
            result = self.fetch_realtime_data_sequential(
                low_priority_providers, symbol, timeout
            )
            if result and result.is_valid():
                return result
        
        logger.error(f"混合获取实时数据策略失败: {symbol}")
        return None
    
    def fetch_realtime_data_parallel(self, 
                                   providers: List[Any], 
                                   symbol: str,
                                   timeout: int = 5) -> Optional[FetchResult]:
        """并行获取实时数据"""
        if not providers:
            return None
        
        logger.info(f"开始并行获取 {symbol} 实时数据，使用 {len(providers)} 个数据源")
        
        # 使用线程池并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_provider = {}
            for provider in providers:
                future = executor.submit(
                    self._fetch_single_realtime_source, 
                    provider, symbol
                )
                future_to_provider[future] = provider
            
            # 等待第一个成功的结果
            try:
                for future in concurrent.futures.as_completed(future_to_provider, timeout=timeout):
                    provider = future_to_provider[future]
                    try:
                        result = future.result()
                        if result.is_valid():
                            logger.info(f"并行获取实时数据成功: {result.source}, 响应时间: {result.response_time:.2f}秒")
                            # 取消其他未完成的任务
                            for f in future_to_provider:
                                if not f.done():
                                    f.cancel()
                            return result
                        else:
                            logger.warning(f"实时数据源 {result.source} 返回无效数据: {result.error}")
                    except Exception as e:
                        logger.warning(f"实时数据源 {provider.name} 执行异常: {e}")
                        continue
            
            except concurrent.futures.TimeoutError:
                logger.error(f"并行获取实时数据超时 ({timeout}秒)")
                # 取消所有未完成的任务
                for future in future_to_provider:
                    future.cancel()
        
        logger.error(f"所有实时数据源并行获取均失败: {symbol}")
        return None
    
    def fetch_realtime_data_sequential(self, 
                                     providers: List[Any], 
                                     symbol: str,
                                     timeout: int = 5) -> Optional[FetchResult]:
        """串行获取实时数据"""
        if not providers:
            return None
        
        logger.info(f"开始串行获取 {symbol} 实时数据")
        
        for provider in providers:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._fetch_single_realtime_source, 
                    provider, symbol
                )
                
                try:
                    result = future.result(timeout=timeout)
                    if result.is_valid():
                        logger.info(f"串行获取实时数据成功: {result.source}, 响应时间: {result.response_time:.2f}秒")
                        return result
                    else:
                        logger.warning(f"实时数据源 {result.source} 返回无效数据: {result.error}")
                        continue
                        
                except concurrent.futures.TimeoutError:
                    logger.warning(f"实时数据源 {provider.name} 超时 ({timeout}秒)")
                    future.cancel()
                    continue
                except Exception as e:
                    logger.warning(f"实时数据源 {provider.name} 异常: {e}")
                    continue
        
        logger.error(f"所有实时数据源串行获取均失败: {symbol}")
        return None
    
    def _fetch_single_realtime_source(self, 
                                    provider: Any, 
                                    symbol: str) -> FetchResult:
        """从单个数据源获取实时数据"""
        start_time = time.time()
        
        try:
            logger.debug(f"开始从 {provider.name} 获取实时数据")
            data = provider.get_realtime_data(symbol)
            response_time = time.time() - start_time
            
            return FetchResult(
                success=True,
                data=data,
                source=provider.name,
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.debug(f"实时数据源 {provider.name} 获取失败: {e}")
            return FetchResult(
                success=False,
                data=None,
                source=provider.name,
                response_time=response_time,
                error=str(e)
            )
    
    def benchmark_providers(self, 
                          providers: List[Any], 
                          symbol: str = "AAPL", 
                          days: int = 7) -> Dict[str, Dict[str, Any]]:
        """基准测试数据源性能"""
        from datetime import datetime, timedelta
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        results = {}
        
        for provider in providers:
            logger.info(f"基准测试: {provider.name}")
            result = self._fetch_single_source(provider, symbol, start_date, end_date)
            
            results[provider.name] = {
                'success': result.success,
                'response_time': result.response_time,
                'data_points': len(result.data) if result.data is not None else 0,
                'error': result.error,
                'health_status': getattr(provider.health, 'status', 'unknown')
            }
        
        return results

# 全局并发获取器实例
_global_concurrent_fetcher: Optional[ConcurrentDataFetcher] = None

def get_concurrent_fetcher(max_workers: int = 3, timeout: int = 10) -> ConcurrentDataFetcher:
    """获取全局并发获取器实例"""
    global _global_concurrent_fetcher
    if _global_concurrent_fetcher is None:
        _global_concurrent_fetcher = ConcurrentDataFetcher(max_workers, timeout)
    return _global_concurrent_fetcher