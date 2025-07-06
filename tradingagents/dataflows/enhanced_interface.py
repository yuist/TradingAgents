#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的数据接口
集成多数据源管理器，提供统一的数据获取接口
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from ..utils.logging_manager import get_logger

# 导入多数据源管理器
try:
    from .multi_source_manager import (
        MultiSourceDataManager, 
        get_multi_source_manager,
        get_stock_data_with_fallback
    )
    from .data_source_config import get_default_config
except ImportError:
    # 如果导入失败，提供占位符
    def get_stock_data_with_fallback(*args, **kwargs):
        raise ImportError("多数据源管理器未正确安装")
    
    def get_default_config():
        return {}

# 导入原有的数据获取函数作为备用
try:
    from .interface import get_YFin_data_online as _original_yahoo_finance
except ImportError:
    def _original_yahoo_finance(*args, **kwargs):
        raise ImportError("原始Yahoo Finance接口不可用")

logger = get_logger('dataflow', 'enhanced_interface')

class EnhancedDataInterface:
    """增强的数据接口类"""
    
    def __init__(self, config: Dict[str, Any] = None, enable_fallback: bool = True):
        """
        初始化增强数据接口
        
        Args:
            config: 数据源配置
            enable_fallback: 是否启用原始接口作为最后备用
        """
        self.config = config or get_default_config()
        self.enable_fallback = enable_fallback
        self.manager = None
        self._initialize_manager()
    
    def _initialize_manager(self):
        """初始化多数据源管理器"""
        try:
            logger.info(
                "Starting multi-source data manager initialization",
                extra={'component': 'enhanced_interface', 'action': 'initialize_manager'}
            )
            # 设置初始化超时
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("多数据源管理器初始化超时")
            
            # 在Windows上使用线程超时而不是signal
            import threading
            import time
            
            result = [None]
            exception = [None]
            
            def init_worker():
                try:
                    result[0] = get_multi_source_manager(self.config)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=init_worker)
            thread.daemon = True
            thread.start()
            thread.join(timeout=30)  # 30秒超时
            
            if thread.is_alive():
                logger.error(
                    "Multi-source data manager initialization timeout (30s)",
                    extra={'component': 'enhanced_interface', 'timeout_seconds': 30}
                )
                self.manager = None
            elif exception[0]:
                raise exception[0]
            else:
                self.manager = result[0]
                logger.info(
                    "Multi-source data manager initialized successfully",
                    extra={'component': 'enhanced_interface'}
                )
                
        except Exception as e:
            logger.error(
                "Multi-source data manager initialization failed",
                extra={
                    'component': 'enhanced_interface',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            self.manager = None
            if not self.enable_fallback:
                raise
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str, 
                      force_source: str = None) -> str:
        """
        获取股票历史数据（兼容原接口格式）
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            force_source: 强制使用指定数据源
        
        Returns:
            CSV格式的股票数据字符串
        """
        logger.info(f"开始获取股票数据: {symbol}, {start_date} 到 {end_date}")
        
        try:
            # 尝试使用多数据源管理器
            if self.manager and not force_source:
                logger.info("使用多数据源管理器获取数据")
                
                # 使用线程超时控制
                import threading
                result = [None]
                exception = [None]
                
                def get_data_worker():
                    try:
                        result[0] = self.manager.get_stock_data(symbol, start_date, end_date)
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=get_data_worker)
                thread.daemon = True
                thread.start()
                thread.join(timeout=60)  # 60秒超时
                
                if thread.is_alive():
                    logger.error("多数据源管理器获取数据超时，切换到备用接口")
                elif exception[0]:
                    raise exception[0]
                elif result[0] is not None and not result[0].empty:
                    return self._dataframe_to_csv_string(result[0], symbol)
                else:
                    logger.warning("多数据源管理器返回空数据")
            
            # 如果指定了数据源，尝试使用特定数据源
            elif force_source:
                return self._get_data_from_specific_source(symbol, start_date, end_date, force_source)
            
            # 最后备用：使用原始Yahoo Finance接口
            if self.enable_fallback:
                logger.warning("使用原始Yahoo Finance接口作为备用")
                return _original_yahoo_finance(symbol, start_date, end_date)
            
            else:
                raise Exception("所有数据源均不可用")
                
        except Exception as e:
            logger.error(f"获取股票数据失败 {symbol}: {e}")
            
            # 最后的备用尝试
            if self.enable_fallback and not force_source:
                try:
                    logger.info("尝试使用原始Yahoo Finance接口")
                    return _original_yahoo_finance(symbol, start_date, end_date)
                except Exception as fallback_error:
                    logger.error(f"备用接口也失败: {fallback_error}")
            
            return f"未找到股票代码 {symbol} 的数据"
    
    def _get_data_from_specific_source(self, symbol: str, start_date: str, 
                                     end_date: str, source_name: str) -> str:
        """从指定数据源获取数据"""
        if not self.manager:
            raise Exception("多数据源管理器未初始化")
        
        # 获取指定数据源的提供者
        for market, providers in self.manager.providers.items():
            for provider in providers:
                if provider.name.lower().replace(' ', '_') == source_name.lower():
                    df = provider.get_stock_data(symbol, start_date, end_date)
                    return self._dataframe_to_csv_string(df, symbol)
        
        raise ValueError(f"未找到数据源: {source_name}")
    
    def _dataframe_to_csv_string(self, df: pd.DataFrame, symbol: str) -> str:
        """将DataFrame转换为CSV字符串格式（兼容原接口）"""
        if df.empty:
            return f"未找到股票代码 {symbol} 的数据"
        
        # 添加注释头
        header = f"# 股票代码: {symbol}\n"
        header += f"# 数据获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"# 数据条数: {len(df)}\n"
        
        # 转换为CSV格式
        csv_content = df.to_csv()
        
        return header + csv_content
    
    def get_dataframe(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取股票数据的DataFrame格式
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            股票数据DataFrame
        """
        if self.manager:
            return self.manager.get_stock_data(symbol, start_date, end_date)
        else:
            # 解析原始接口返回的CSV数据
            csv_data = self.get_stock_data(symbol, start_date, end_date)
            if csv_data.startswith("未找到"):
                return pd.DataFrame()
            
            # 解析CSV数据
            lines = csv_data.split('\n')
            data_lines = [line for line in lines if line and not line.startswith('#')]
            
            if len(data_lines) > 1:
                return pd.read_csv(pd.StringIO('\n'.join(data_lines)), index_col=0, parse_dates=True)
            
            return pd.DataFrame()
    
    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取实时数据
        
        Args:
            symbol: 股票代码
        
        Returns:
            实时数据字典
        """
        if self.manager:
            return self.manager.get_realtime_data(symbol)
        else:
            raise NotImplementedError("实时数据需要多数据源管理器支持")
    
    def get_available_sources(self) -> Dict[str, List[str]]:
        """
        获取可用的数据源列表
        
        Returns:
            按市场类型分组的数据源列表
        """
        if self.manager:
            return self.manager._get_provider_summary()
        else:
            return {"fallback": ["Yahoo Finance (原始接口)"]}
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        获取数据源健康状态
        
        Returns:
            健康状态信息
        """
        if self.manager:
            return self.manager.get_health_status()
        else:
            return {"status": "仅备用接口可用"}
    
    def force_health_check(self):
        """强制执行健康检查"""
        if self.manager:
            self.manager.force_health_check()
        else:
            logger.warning("多数据源管理器未初始化，无法执行健康检查")
    
    def switch_to_source(self, source_name: str, symbol: str, start_date: str, end_date: str) -> str:
        """
        切换到指定数据源获取数据
        
        Args:
            source_name: 数据源名称
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            CSV格式数据
        """
        return self.get_stock_data(symbol, start_date, end_date, force_source=source_name)
    
    def benchmark_sources(self, symbol: str = "AAPL", days: int = 7) -> Dict[str, Dict[str, Any]]:
        """
        对比不同数据源的性能
        
        Args:
            symbol: 测试用股票代码
            days: 测试数据天数
        
        Returns:
            性能对比结果
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        results = {}
        
        if self.manager:
            for market, providers in self.manager.providers.items():
                for provider in providers:
                    try:
                        start_time = datetime.now()
                        data = provider.get_stock_data(symbol, start_date, end_date)
                        end_time = datetime.now()
                        
                        results[provider.name] = {
                            'success': True,
                            'response_time': (end_time - start_time).total_seconds(),
                            'data_points': len(data) if not data.empty else 0,
                            'health_status': provider.health.status.value
                        }
                    except Exception as e:
                        results[provider.name] = {
                            'success': False,
                            'error': str(e),
                            'health_status': provider.health.status.value
                        }
        
        # 测试备用接口
        if self.enable_fallback:
            try:
                start_time = datetime.now()
                data = _original_yahoo_finance(symbol, start_date, end_date)
                end_time = datetime.now()
                
                results['Yahoo Finance (原始)'] = {
                    'success': not data.startswith("未找到"),
                    'response_time': (end_time - start_time).total_seconds(),
                    'data_format': 'CSV字符串'
                }
            except Exception as e:
                results['Yahoo Finance (原始)'] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results

# 全局实例
_global_interface: Optional[EnhancedDataInterface] = None

def get_enhanced_interface(config: Dict[str, Any] = None, 
                          enable_fallback: bool = True) -> EnhancedDataInterface:
    """获取全局增强数据接口实例"""
    global _global_interface
    
    if _global_interface is None:
        _global_interface = EnhancedDataInterface(config, enable_fallback)
    
    return _global_interface

# 兼容性函数（替换原有的get_YFin_data_online）
def get_YFin_data_online_enhanced(symbol: str, start_date: str, end_date: str) -> str:
    """
    增强版的Yahoo Finance数据获取函数
    兼容原接口，但支持多数据源备用
    
    Args:
        symbol: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        CSV格式的股票数据字符串
    """
    interface = get_enhanced_interface()
    return interface.get_stock_data(symbol, start_date, end_date)

# 便捷函数
def get_stock_dataframe(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取股票数据DataFrame"""
    interface = get_enhanced_interface()
    return interface.get_dataframe(symbol, start_date, end_date)

def get_realtime_quote(symbol: str) -> Dict[str, Any]:
    """获取实时报价"""
    interface = get_enhanced_interface()
    return interface.get_realtime_data(symbol)

def check_data_sources_health() -> Dict[str, Any]:
    """检查数据源健康状态"""
    interface = get_enhanced_interface()
    return interface.get_health_status()

def benchmark_data_sources(symbol: str = "AAPL") -> Dict[str, Dict[str, Any]]:
    """对比数据源性能"""
    interface = get_enhanced_interface()
    return interface.benchmark_sources(symbol)

if __name__ == "__main__":
    # 示例用法
    print("增强数据接口测试")
    print("=" * 50)
    
    # 创建接口实例
    interface = get_enhanced_interface()
    
    # 显示可用数据源
    sources = interface.get_available_sources()
    print(f"可用数据源: {sources}")
    
    # 健康检查
    print("\n执行健康检查...")
    interface.force_health_check()
    
    health = interface.get_health_status()
    print(f"健康状态: {health}")
    
    # 测试数据获取
    print("\n测试数据获取...")
    try:
        data = interface.get_stock_data("AAPL", "2024-01-01", "2024-01-07")
        print(f"数据获取成功，长度: {len(data)} 字符")
        print(f"前200字符: {data[:200]}...")
    except Exception as e:
        print(f"数据获取失败: {e}")
    
    # 性能对比
    print("\n性能对比测试...")
    benchmark = interface.benchmark_sources("AAPL", 3)
    for source, result in benchmark.items():
        print(f"{source}: {result}")
    
    print("\n测试完成！")