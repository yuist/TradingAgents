#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多数据源管理器
实现智能的数据源切换和故障恢复机制
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from ..utils.logging_manager import get_logger

# 导入新的优化组件
from .circuit_breaker import get_circuit_breaker_manager
from .data_cache import get_data_cache
from .concurrent_fetcher import get_concurrent_fetcher

# 导入现有的数据获取函数
try:
    from .interface import get_YFin_data_online
    from .alpha_vantage_utils import get_alpha_vantage_data
    from .polygon_utils import get_polygon_data
except ImportError:
    # 如果导入失败，定义占位符函数
    def get_YFin_data_online(*args, **kwargs):
        raise NotImplementedError("Yahoo Finance provider not available")
    
def get_alpha_vantage_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Alpha Vantage 数据获取函数"""
    from .alpha_vantage_utils import get_stock_data_alpha_vantage
    
    result = get_stock_data_alpha_vantage(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    if result is not None and not result.empty:
        return result
    
    raise ValueError(f"Alpha Vantage 未返回有效数据: {symbol}")

def get_polygon_data(*args, **kwargs):
    raise NotImplementedError("Polygon provider not available")

logger = get_logger('dataflow', 'multi_source_manager')

class MarketType(Enum):
    """市场类型枚举"""
    DOMESTIC = "domestic"  # 国内市场
    INTERNATIONAL = "international"  # 国际市场
    CRYPTO = "crypto"  # 加密货币

class DataSourceStatus(Enum):
    """数据源状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"

@dataclass
class DataSourceHealth:
    """数据源健康状态"""
    status: DataSourceStatus
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    failure_count: int
    avg_response_time: float
    success_rate: float

class DataSourceInterface(ABC):
    """数据源接口抽象类"""
    
    def __init__(self, name: str, priority: int = 1):
        self.name = name
        self.priority = priority
        self.health = DataSourceHealth(
            status=DataSourceStatus.UNKNOWN,
            last_success=None,
            last_failure=None,
            failure_count=0,
            avg_response_time=0.0,
            success_rate=0.0
        )
    
    @abstractmethod
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票历史数据"""
        pass
    
    @abstractmethod
    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """获取实时数据"""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """健康检查"""
        pass
    
    @abstractmethod
    def get_supported_markets(self) -> List[MarketType]:
        """获取支持的市场类型"""
        pass
    
    def update_health_success(self, response_time: float):
        """更新成功状态"""
        self.health.last_success = datetime.now()
        self.health.failure_count = 0
        self.health.status = DataSourceStatus.HEALTHY
        
        # 更新平均响应时间
        if self.health.avg_response_time == 0:
            self.health.avg_response_time = response_time
        else:
            self.health.avg_response_time = (self.health.avg_response_time + response_time) / 2
    
    def update_health_failure(self, error: Exception):
        """更新失败状态"""
        self.health.last_failure = datetime.now()
        self.health.failure_count += 1
        
        if self.health.failure_count >= 3:
            self.health.status = DataSourceStatus.FAILED
        else:
            self.health.status = DataSourceStatus.DEGRADED
        
        logger.warning(
            f"Data source {self.name} failed",
            extra={
                'data_source': self.name,
                'failure_count': self.health.failure_count,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'component': 'multi_source_manager'
            }
        )

class YahooFinanceProvider(DataSourceInterface):
    """Yahoo Finance 数据提供者"""
    
    def __init__(self):
        super().__init__("Yahoo Finance", priority=2)
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            start_time = time.time()
            result = get_YFin_data_online(symbol, start_date, end_date)
            response_time = time.time() - start_time
            
            if result and not result.startswith("未找到"):
                # 解析CSV数据为DataFrame
                lines = result.split('\n')
                data_lines = [line for line in lines if line and not line.startswith('#')]
                
                if len(data_lines) > 1:  # 至少有标题行和一行数据
                    df = pd.read_csv(pd.StringIO('\n'.join(data_lines)), index_col=0, parse_dates=True)
                    self.update_health_success(response_time)
                    return df
            
            raise ValueError(f"无效的数据: {result}")
            
        except Exception as e:
            self.update_health_failure(e)
            raise
    
    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """获取实时数据"""
        # Yahoo Finance 实时数据实现
        raise NotImplementedError("Yahoo Finance 实时数据获取待实现")
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            # 使用一个简单的请求测试连接
            test_data = self.get_stock_data("AAPL", "2024-01-01", "2024-01-02")
            return not test_data.empty
        except:
            return False
    
    def get_supported_markets(self) -> List[MarketType]:
        return [MarketType.DOMESTIC, MarketType.INTERNATIONAL]

class AlphaVantageProvider(DataSourceInterface):
    """Alpha Vantage 数据提供者"""
    
    def __init__(self, api_key: str):
        super().__init__("Alpha Vantage", priority=1)
        self.api_key = api_key
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            start_time = time.time()
            
            # 使用alpha_vantage_utils中的函数
            from tradingagents.dataflows.alpha_vantage_utils import get_stock_data_alpha_vantage
            
            result = get_stock_data_alpha_vantage(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            response_time = time.time() - start_time
            
            if result is not None and not result.empty:
                self.update_health_success(response_time)
                return result
            
            raise ValueError(f"Alpha Vantage 未返回有效数据: {symbol}")
            
        except Exception as e:
            self.update_health_failure(e)
            raise
    
    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """获取实时数据"""
        raise NotImplementedError("Alpha Vantage 实时数据获取待实现")
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            test_data = self.get_stock_data("AAPL", "2024-01-01", "2024-01-02")
            return not test_data.empty
        except:
            return False
    
    def get_supported_markets(self) -> List[MarketType]:
        return [MarketType.INTERNATIONAL]

class TuShareProvider(DataSourceInterface):
    """TuShare 数据提供者（国内股票）"""
    
    def __init__(self, token: str):
        super().__init__("TuShare Pro", priority=1)
        self.token = token
        self._tushare = None
    
    def _get_tushare(self):
        """延迟加载 TuShare"""
        if self._tushare is None:
            try:
                import tushare as ts
                ts.set_token(self.token)
                self._tushare = ts.pro_api()
            except ImportError:
                raise ImportError("TuShare 未安装，请运行: pip install tushare")
        return self._tushare
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            start_time = time.time()
            ts_api = self._get_tushare()
            
            # 转换股票代码格式
            ts_code = self._convert_symbol_to_tushare(symbol)
            
            # 获取日线数据
            df = ts_api.daily(
                ts_code=ts_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )
            
            if not df.empty:
                # 转换为标准格式
                df = self._convert_to_standard_format(df)
                response_time = time.time() - start_time
                self.update_health_success(response_time)
                return df
            
            raise ValueError(f"TuShare 未返回数据: {symbol}")
            
        except Exception as e:
            self.update_health_failure(e)
            raise
    
    def _convert_symbol_to_tushare(self, symbol: str) -> str:
        """转换股票代码为 TuShare 格式"""
        # 简单的转换逻辑，实际需要更复杂的处理
        if symbol.endswith('.SH') or symbol.endswith('.SZ'):
            return symbol
        elif symbol.startswith('6'):
            return f"{symbol}.SH"
        elif symbol.startswith('0') or symbol.startswith('3'):
            return f"{symbol}.SZ"
        else:
            return f"{symbol}.SH"  # 默认上海
    
    def _convert_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换为标准格式"""
        # 重命名列
        column_mapping = {
            'trade_date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'vol': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.sort_index()
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """获取实时数据"""
        try:
            start_time = time.time()
            ts_api = self._get_tushare()
            
            # 转换股票代码格式
            ts_code = self._convert_symbol_to_tushare(symbol)
            
            # 获取实时行情数据
            df = ts_api.query('daily_basic', ts_code=ts_code, trade_date='')
            
            if not df.empty:
                latest_data = df.iloc[0]
                response_time = time.time() - start_time
                
                # 更新健康状态
                self.health.last_success = time.time()
                self.health.response_time = response_time
                self.health.status = DataSourceStatus.HEALTHY
                
                # 转换为标准格式
                realtime_data = {
                    'symbol': symbol,
                    'price': float(latest_data.get('close', 0)),
                    'volume': int(latest_data.get('vol', 0)),
                    'timestamp': time.time(),
                    'source': self.name
                }
                
                logger.debug(f"TuShare 实时数据获取成功: {symbol}")
                return realtime_data
            else:
                raise ValueError(f"未找到 {symbol} 的实时数据")
                
        except Exception as e:
            response_time = time.time() - start_time
            self.health.last_error = str(e)
            self.health.response_time = response_time
            self.health.error_count += 1
            
            if self.health.error_count >= 3:
                self.health.status = DataSourceStatus.FAILED
            
            logger.error(f"TuShare 实时数据获取失败: {symbol}, 错误: {e}")
            raise
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            test_data = self.get_stock_data("000001", "2024-01-01", "2024-01-02")
            return not test_data.empty
        except:
            return False
    
    def get_supported_markets(self) -> List[MarketType]:
        return [MarketType.DOMESTIC]

class AkShareProvider(DataSourceInterface):
    """AkShare 数据提供者"""
    
    def __init__(self):
        super().__init__("AkShare", priority=1)
        self._akshare = None
    
    def _get_akshare(self):
        """延迟加载 AkShare"""
        if self._akshare is None:
            try:
                import akshare as ak
                self._akshare = ak
            except ImportError:
                raise ImportError("AkShare 未安装，请运行: pip install akshare")
        return self._akshare
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            start_time = time.time()
            ak = self._get_akshare()
            
            # 转换股票代码格式
            ak_symbol = self._convert_symbol_to_akshare(symbol)
            
            # 获取股票历史数据
            df = ak.stock_zh_a_hist(
                symbol=ak_symbol,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust=""
            )
            
            if not df.empty:
                # 转换为标准格式
                df = self._convert_to_standard_format(df)
                response_time = time.time() - start_time
                self.update_health_success(response_time)
                return df
            
            raise ValueError(f"AkShare 未返回数据: {symbol}")
            
        except Exception as e:
            self.update_health_failure(e)
            raise
    
    def _convert_symbol_to_akshare(self, symbol: str) -> str:
        """转换股票代码为 AkShare 格式"""
        # 移除后缀
        if symbol.endswith('.SH') or symbol.endswith('.SZ'):
            return symbol[:-3]
        elif symbol.endswith('.sz') or symbol.endswith('.sh'):
            return symbol[:-3]
        return symbol
    
    def _convert_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换为标准格式"""
        # 重命名列
        column_mapping = {
            '日期': 'Date',
            '开盘': 'Open',
            '最高': 'High',
            '最低': 'Low',
            '收盘': 'Close',
            '成交量': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.sort_index()
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """获取实时数据"""
        try:
            start_time = time.time()
            ak = self._get_akshare()
            
            # 转换股票代码格式
            ak_symbol = self._convert_symbol_to_akshare(symbol)
            
            # 获取实时行情数据
            df = ak.stock_zh_a_spot_em()
            
            # 查找对应股票
            stock_data = df[df['代码'] == ak_symbol]
            
            if not stock_data.empty:
                latest_data = stock_data.iloc[0]
                response_time = time.time() - start_time
                self.update_health_success(response_time)
                
                # 转换为标准格式
                realtime_data = {
                    'symbol': symbol,
                    'price': float(latest_data.get('最新价', 0)),
                    'volume': int(latest_data.get('成交量', 0)),
                    'timestamp': time.time(),
                    'source': self.name
                }
                
                return realtime_data
            else:
                raise ValueError(f"未找到 {symbol} 的实时数据")
                
        except Exception as e:
            self.update_health_failure(e)
            raise
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            test_data = self.get_stock_data("000001", "2024-01-01", "2024-01-02")
            return not test_data.empty
        except:
            return False
    
    def get_supported_markets(self) -> List[MarketType]:
        return [MarketType.DOMESTIC]

class EfinanceProvider(DataSourceInterface):
    """efinance 数据提供者"""
    
    def __init__(self):
        super().__init__("efinance", priority=2)
        self._efinance = None
    
    def _get_efinance(self):
        """延迟加载 efinance"""
        if self._efinance is None:
            try:
                import efinance as ef
                self._efinance = ef
            except ImportError:
                raise ImportError("efinance 未安装，请运行: pip install efinance")
        return self._efinance
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            start_time = time.time()
            ef = self._get_efinance()
            
            # 转换股票代码格式
            ef_symbol = self._convert_symbol_to_efinance(symbol)
            
            # 获取股票历史数据
            df = ef.stock.get_quote_history(
                stock_codes=ef_symbol,
                beg=start_date,
                end=end_date
            )
            
            if not df.empty:
                # 转换为标准格式
                df = self._convert_to_standard_format(df)
                response_time = time.time() - start_time
                self.update_health_success(response_time)
                return df
            
            raise ValueError(f"efinance 未返回数据: {symbol}")
            
        except Exception as e:
            self.update_health_failure(e)
            raise
    
    def _convert_symbol_to_efinance(self, symbol: str) -> str:
        """转换股票代码为 efinance 格式"""
        # 移除后缀
        if symbol.endswith('.SH') or symbol.endswith('.SZ'):
            return symbol[:-3]
        elif symbol.endswith('.sz') or symbol.endswith('.sh'):
            return symbol[:-3]
        return symbol
    
    def _convert_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换为标准格式"""
        # 重命名列
        column_mapping = {
            '日期': 'Date',
            '开盘': 'Open',
            '最高': 'High',
            '最低': 'Low',
            '收盘': 'Close',
            '成交量': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.sort_index()
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """获取实时数据"""
        try:
            start_time = time.time()
            ef = self._get_efinance()
            
            # 转换股票代码格式
            ef_symbol = self._convert_symbol_to_efinance(symbol)
            
            # 获取实时行情数据
            df = ef.stock.get_realtime_quotes([ef_symbol])
            
            if not df.empty:
                latest_data = df.iloc[0]
                response_time = time.time() - start_time
                self.update_health_success(response_time)
                
                # 转换为标准格式
                realtime_data = {
                    'symbol': symbol,
                    'price': float(latest_data.get('最新价', 0)),
                    'volume': int(latest_data.get('成交量', 0)),
                    'timestamp': time.time(),
                    'source': self.name
                }
                
                return realtime_data
            else:
                raise ValueError(f"未找到 {symbol} 的实时数据")
                
        except Exception as e:
            self.update_health_failure(e)
            raise
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            test_data = self.get_stock_data("000001", "2024-01-01", "2024-01-02")
            return not test_data.empty
        except:
            return False
    
    def get_supported_markets(self) -> List[MarketType]:
        return [MarketType.DOMESTIC]

class AdataProvider(DataSourceInterface):
    """adata 数据提供者"""
    
    def __init__(self):
        super().__init__("adata", priority=3)
        self._adata = None
    
    def _get_adata(self):
        """延迟加载 adata"""
        if self._adata is None:
            try:
                import adata
                self._adata = adata
            except ImportError:
                raise ImportError("adata 未安装，请运行: pip install adata")
        return self._adata
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            start_time = time.time()
            adata = self._get_adata()
            
            # 转换股票代码格式
            adata_symbol = self._convert_symbol_to_adata(symbol)
            
            # 获取股票历史数据
            df = adata.stock.market.get_market(
                stock_code=adata_symbol,
                k_type=1,  # 日K线
                start_date=start_date
            )
            
            if not df.empty:
                # 转换为标准格式
                df = self._convert_to_standard_format(df)
                response_time = time.time() - start_time
                self.update_health_success(response_time)
                return df
            
            raise ValueError(f"adata 未返回数据: {symbol}")
            
        except Exception as e:
            self.update_health_failure(e)
            raise
    
    def _convert_symbol_to_adata(self, symbol: str) -> str:
        """转换股票代码为 adata 格式"""
        # 移除后缀
        if symbol.endswith('.SH') or symbol.endswith('.SZ'):
            return symbol[:-3]
        elif symbol.endswith('.sz') or symbol.endswith('.sh'):
            return symbol[:-3]
        return symbol
    
    def _convert_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换为标准格式"""
        # 重命名列
        column_mapping = {
            'trade_time': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.sort_index()
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """获取实时数据"""
        # adata 主要用于历史数据，实时数据功能有限
        raise NotImplementedError("adata 实时数据获取待实现")
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            test_data = self.get_stock_data("000001", "2024-01-01", "2024-01-02")
            return not test_data.empty
        except:
            return False
    
    def get_supported_markets(self) -> List[MarketType]:
        return [MarketType.DOMESTIC]

class SinaFinanceProvider(DataSourceInterface):
    """新浪财经数据提供者"""
    
    def __init__(self):
        super().__init__("新浪财经", priority=2)
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            start_time = time.time()
            import requests
            import json
            from datetime import datetime, timedelta
            
            # 转换股票代码
            sina_symbol = self._convert_symbol_to_sina(symbol)
            
            # 计算需要获取的数据长度
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days_diff = (end_dt - start_dt).days + 50  # 多获取一些数据以确保覆盖
            datalen = min(days_diff, 1023)  # 新浪API限制
            
            # 构建API URL - 使用新浪财经的历史数据接口
            url = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
            params = {
                'symbol': sina_symbol,
                'scale': 240,  # 日线数据
                'ma': 'no',
                'datalen': datalen
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'http://finance.sina.com.cn/'
            }
            
            proxies = {
                'http': None,
                'https': None
            }
            response = requests.get(url, params=params, headers=headers, timeout=15, proxies=proxies)
            response.raise_for_status()

            # 新浪财经的历史数据接口可能使用gbk编码
            response.encoding = 'gbk'
            
            # 解析JSON数据
            response_text = response.text
            if not response_text or response_text.strip() == 'null':
                raise ValueError(f"新浪财经返回空数据: {symbol}")
            
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                raise ValueError(f"新浪财经返回无效JSON数据: {symbol}")
            
            if not data or not isinstance(data, list):
                raise ValueError(f"新浪财经返回无效数据格式: {symbol}")
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            if df.empty:
                raise ValueError(f"新浪财经返回空数据集: {symbol}")
            
            # 数据格式转换和清理
            df = self._convert_sina_to_standard_format(df)
            
            # 过滤日期范围
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            if df.empty:
                raise ValueError(f"指定日期范围内无数据: {symbol} ({start_date} 到 {end_date})")
            
            response_time = time.time() - start_time
            self.update_health_success(response_time)
            
            logger.debug(f"新浪财经数据获取成功: {symbol}, 数据行数: {len(df)}")
            return df
                
        except Exception as e:
            self.update_health_failure(e)
            raise
    
    def _convert_symbol_to_sina(self, symbol: str) -> str:
        """转换股票代码为新浪格式"""
        # 移除可能的后缀（大小写不敏感）
        clean_symbol = symbol.upper().replace('.SH', '').replace('.SZ', '').replace('.SS', '')
        
        if clean_symbol.startswith('6'):  # 上海股票
            return f"sh{clean_symbol}"
        elif clean_symbol.startswith('0') or clean_symbol.startswith('3'):  # 深圳股票
            return f"sz{clean_symbol}"
        elif clean_symbol.startswith('8') or clean_symbol.startswith('4'):  # 北交所
            return f"bj{clean_symbol}"
        else:
            # 对于美股等其他市场，直接返回原始代码
            return symbol
    
    def _convert_sina_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """将新浪财经数据转换为标准格式"""
        try:
            # 新浪财经返回的字段：day, open, high, low, close, volume
            required_columns = ['day', 'open', 'high', 'low', 'close', 'volume']
            
            # 检查必需的列是否存在
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"缺少必需的列: {missing_columns}")
            
            # 重命名列为标准格式
            df = df.rename(columns={
                'day': 'Date',
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # 转换数据类型
            df['Date'] = pd.to_datetime(df['Date'])
            df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
            df['High'] = pd.to_numeric(df['High'], errors='coerce')
            df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            # 添加调整后收盘价（与收盘价相同）
            df['Adj Close'] = df['Close']
            
            # 设置日期为索引
            df = df.set_index('Date')
            
            # 按日期排序
            df = df.sort_index()
            
            # 移除包含NaN的行
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"新浪财经数据格式转换失败: {e}")
            raise ValueError(f"数据格式转换失败: {e}")
    

    
    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """获取实时数据"""
        try:
            start_time = time.time()
            import requests
            
            # 转换股票代码
            sina_symbol = self._convert_symbol_to_sina(symbol)
            
            # 新浪实时数据API
            url = f"http://hq.sinajs.cn/list={sina_symbol}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'http://finance.sina.com.cn/'
            }
            
            proxies = {
                'http': None,
                'https': None
            }
            response = requests.get(url, headers=headers, timeout=10, proxies=proxies)
            response.raise_for_status()
            
            # 新浪财经使用GBK编码
            response.encoding = 'gbk'
            content = response.text
            if content and '=' in content:
                try:
                    data_str = content.split('="')[1].split('";')[0]
                    data_parts = data_str.split(',')
                    
                    if len(data_parts) >= 32 and data_parts[0]:  # 确保有股票名称
                        response_time = time.time() - start_time
                        self.update_health_success(response_time)
                        
                        # 解析新浪财经实时数据格式
                        # 0:股票名称, 1:今开, 2:昨收, 3:现价, 4:最高, 5:最低, 8:成交量, 9:成交额
                        current_price = float(data_parts[3]) if data_parts[3] else 0.0
                        volume = int(float(data_parts[8])) if data_parts[8] else 0
                        
                        realtime_data = {
                            'symbol': symbol,
                            'name': data_parts[0],
                            'price': current_price,
                            'open': float(data_parts[1]) if data_parts[1] else 0.0,
                            'prev_close': float(data_parts[2]) if data_parts[2] else 0.0,
                            'high': float(data_parts[4]) if data_parts[4] else 0.0,
                            'low': float(data_parts[5]) if data_parts[5] else 0.0,
                            'volume': volume,
                            'amount': float(data_parts[9]) if data_parts[9] else 0.0,
                            'timestamp': time.time(),
                            'source': self.name
                        }
                        
                        logger.debug(f"新浪财经实时数据获取成功: {symbol}, 价格: {current_price}")
                        return realtime_data
                    else:
                        raise ValueError(f"实时数据格式不正确或股票代码无效: {symbol}")
                except (IndexError, ValueError) as e:
                    raise ValueError(f"实时数据解析失败: {symbol}, 错误: {e}")
            else:
                raise ValueError(f"未找到 {symbol} 的实时数据")
                
        except Exception as e:
            self.update_health_failure(e)
            raise
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            realtime_data = self.get_realtime_data("000001")
            return realtime_data is not None
        except:
            return False
    
    def get_supported_markets(self) -> List[MarketType]:
        return [MarketType.DOMESTIC]

class MultiSourceDataManager:
    """多数据源管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.providers: Dict[MarketType, List[DataSourceInterface]] = {
            MarketType.DOMESTIC: [],
            MarketType.INTERNATIONAL: [],
            MarketType.CRYPTO: []
        }
        self.config = config or {}
        self.health_check_interval = self.config.get('health_check_interval', 300)  # 5分钟
        self.last_health_check = {}
        
        # 初始化数据源
        self._initialize_providers()
    
    def _initialize_providers(self):
        """初始化数据源提供者"""
        try:
            # 获取数据源配置
            data_sources = self.config.get('data_sources', {})
            
            # Yahoo Finance (通用)
            if data_sources.get('yahoo_finance', {}).get('enabled', True):
                try:
                    yahoo_provider = YahooFinanceProvider()
                    for market in yahoo_provider.get_supported_markets():
                        self.providers[market].append(yahoo_provider)
                    logger.info("Yahoo Finance 提供者初始化成功")
                except Exception as e:
                    logger.error(f"Yahoo Finance 提供者初始化失败: {e}")
            
            # Alpha Vantage
            alpha_vantage_config = data_sources.get('alpha_vantage', {})
            if alpha_vantage_config.get('enabled', False):
                # 从主配置中获取API密钥
                alpha_vantage_key = self.config.get('api_keys', {}).get('alpha_vantage_api_key')
                if alpha_vantage_key:
                    try:
                        alpha_provider = AlphaVantageProvider(alpha_vantage_key)
                        for market in alpha_provider.get_supported_markets():
                            self.providers[market].append(alpha_provider)
                        logger.info("Alpha Vantage 提供者初始化成功")
                    except Exception as e:
                        logger.error(f"Alpha Vantage 提供者初始化失败: {e}")
                else:
                    logger.warning("Alpha Vantage API密钥未配置，跳过初始化")
            
            # TuShare
            tushare_config = data_sources.get('tushare', {})
            if tushare_config.get('enabled', False):
                tushare_token = tushare_config.get('token')
                if tushare_token:
                    try:
                        tushare_provider = TuShareProvider(tushare_token)
                        for market in tushare_provider.get_supported_markets():
                            self.providers[market].append(tushare_provider)
                        logger.info("TuShare 提供者初始化成功")
                    except Exception as e:
                        logger.error(f"TuShare 提供者初始化失败: {e}")
            
            # AkShare
            akshare_config = data_sources.get('akshare', {})
            if akshare_config.get('enabled', False):
                try:
                    akshare_provider = AkShareProvider()
                    for market in akshare_provider.get_supported_markets():
                        self.providers[market].append(akshare_provider)
                    logger.info("AkShare 提供者初始化成功")
                except Exception as e:
                    logger.error(f"AkShare 提供者初始化失败: {e}")
            
            # efinance
            efinance_config = data_sources.get('efinance', {})
            if efinance_config.get('enabled', False):
                try:
                    efinance_provider = EfinanceProvider()
                    for market in efinance_provider.get_supported_markets():
                        self.providers[market].append(efinance_provider)
                    logger.info("efinance 提供者初始化成功")
                except Exception as e:
                    logger.error(f"efinance 提供者初始化失败: {e}")
            
            # adata
            adata_config = data_sources.get('adata', {})
            if adata_config.get('enabled', False):
                try:
                    adata_provider = AdataProvider()
                    for market in adata_provider.get_supported_markets():
                        self.providers[market].append(adata_provider)
                    logger.info("adata 提供者初始化成功")
                except Exception as e:
                    logger.error(f"adata 提供者初始化失败: {e}")
            
            # 新浪财经
            if data_sources.get('sina_finance', {}).get('enabled', True):
                try:
                    sina_provider = SinaFinanceProvider()
                    for market in sina_provider.get_supported_markets():
                        self.providers[market].append(sina_provider)
                    logger.info("新浪财经 提供者初始化成功")
                except Exception as e:
                    logger.error(f"新浪财经 提供者初始化失败: {e}")
            
            # 按优先级排序
            for market in self.providers:
                self.providers[market].sort(key=lambda x: x.priority)
            
            logger.info(f"已初始化数据源: {self._get_provider_summary()}")
            
        except Exception as e:
            logger.error(f"初始化数据源失败: {e}")
    
    def _get_provider_summary(self) -> Dict[str, List[str]]:
        """获取数据源摘要"""
        summary = {}
        for market, providers in self.providers.items():
            summary[market.value] = [p.name for p in providers]
        return summary
    
    def _detect_market_type(self, symbol: str) -> MarketType:
        """检测股票代码的市场类型"""
        # 简单的市场检测逻辑
        if symbol.startswith(('6', '0', '3')) or symbol.endswith(('.SH', '.SZ')):
            return MarketType.DOMESTIC
        else:
            return MarketType.INTERNATIONAL
    
    def _should_perform_health_check(self, provider: DataSourceInterface) -> bool:
        """判断是否需要进行健康检查"""
        last_check = self.last_health_check.get(provider.name)
        if last_check is None:
            return True
        
        return (datetime.now() - last_check).seconds > self.health_check_interval
    
    def _perform_health_check(self, provider: DataSourceInterface):
        """执行健康检查"""
        try:
            if provider.health_check():
                if provider.health.status == DataSourceStatus.FAILED:
                    provider.health.status = DataSourceStatus.HEALTHY
                    provider.health.failure_count = 0
                    logger.info(f"数据源 {provider.name} 已恢复")
            else:
                provider.health.status = DataSourceStatus.DEGRADED
                
        except Exception as e:
            provider.update_health_failure(e)
        
        self.last_health_check[provider.name] = datetime.now()
    
    def _validate_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """验证数据质量"""
        validation_result = self._validate_data_detailed(data, symbol)
        return validation_result['is_valid']
    
    def _validate_data_detailed(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """详细的数据质量验证，返回完整的验证报告"""
        validation_report = {
            'is_valid': True,
            'symbol': symbol,
            'data_points': 0,
            'issues': [],
            'warnings': [],
            'quality_score': 100.0,
            'completeness': 0.0,
            'consistency': 0.0,
            'timeliness': 0.0,
            'outliers': []
        }
        
        try:
            # 基本检查
            if data is None or data.empty:
                validation_report['is_valid'] = False
                validation_report['issues'].append('数据为空')
                validation_report['quality_score'] = 0.0
                return validation_report
            
            validation_report['data_points'] = len(data)
            
            # 1. 数据完整性检查
            completeness_score = self._check_data_completeness(data, validation_report)
            validation_report['completeness'] = completeness_score
            
            # 2. 数据一致性检查
            consistency_score = self._check_data_consistency(data, validation_report)
            validation_report['consistency'] = consistency_score
            
            # 3. 数据时效性检查
            timeliness_score = self._check_data_timeliness(data, validation_report)
            validation_report['timeliness'] = timeliness_score
            
            # 4. 异常值检测
            self._detect_outliers(data, validation_report)
            
            # 5. 计算综合质量分数
            validation_report['quality_score'] = (
                completeness_score * 0.3 +
                consistency_score * 0.4 +
                timeliness_score * 0.2 +
                max(0, 100 - len(validation_report['outliers']) * 5) * 0.1
            )
            
            # 6. 判断是否通过验证
            if validation_report['issues'] or validation_report['quality_score'] < 60:
                validation_report['is_valid'] = False
            
            # 记录验证结果
            if validation_report['is_valid']:
                logger.debug(f"{symbol} 数据验证通过，质量分数: {validation_report['quality_score']:.1f}")
            else:
                logger.warning(f"{symbol} 数据验证失败: {validation_report['issues']}")
            
        except Exception as e:
            validation_report['is_valid'] = False
            validation_report['issues'].append(f'验证过程出错: {str(e)}')
            validation_report['quality_score'] = 0.0
            logger.error(f"数据验证异常: {e}")
        
        return validation_report
    
    def _check_data_completeness(self, data: pd.DataFrame, report: Dict[str, Any]) -> float:
        """检查数据完整性"""
        score = 100.0
        
        # 检查必要的列
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            report['issues'].append(f'缺少必要列: {missing_columns}')
            score -= 30
        
        # 检查数据缺失率
        for col in required_columns:
            if col in data.columns:
                missing_rate = data[col].isna().sum() / len(data)
                if missing_rate > 0.1:  # 超过10%缺失
                    report['issues'].append(f'{col} 列缺失率过高: {missing_rate:.1%}')
                    score -= 20
                elif missing_rate > 0.05:  # 超过5%缺失
                    report['warnings'].append(f'{col} 列有少量缺失: {missing_rate:.1%}')
                    score -= 5
        
        # 检查成交量数据
        if 'Volume' in data.columns:
            zero_volume_rate = (data['Volume'] == 0).sum() / len(data)
            if zero_volume_rate > 0.3:
                report['warnings'].append(f'成交量为零的比例较高: {zero_volume_rate:.1%}')
                score -= 10
        
        return max(0, score)
    
    def _check_data_consistency(self, data: pd.DataFrame, report: Dict[str, Any]) -> float:
        """检查数据一致性"""
        score = 100.0
        
        required_columns = ['Open', 'High', 'Low', 'Close']
        available_columns = [col for col in required_columns if col in data.columns]
        
        if len(available_columns) < 4:
            return score
        
        # 检查价格逻辑关系
        high_low_issues = (data['High'] < data['Low']).sum()
        if high_low_issues > 0:
            report['issues'].append(f'发现 {high_low_issues} 条最高价低于最低价的异常记录')
            score -= 25
        
        # 检查开盘价、收盘价是否在合理范围内
        open_out_of_range = ((data['Open'] > data['High']) | (data['Open'] < data['Low'])).sum()
        close_out_of_range = ((data['Close'] > data['High']) | (data['Close'] < data['Low'])).sum()
        
        if open_out_of_range > 0:
            report['issues'].append(f'发现 {open_out_of_range} 条开盘价超出当日价格范围的记录')
            score -= 15
        
        if close_out_of_range > 0:
            report['issues'].append(f'发现 {close_out_of_range} 条收盘价超出当日价格范围的记录')
            score -= 15
        
        # 检查负价格
        for col in available_columns:
            negative_prices = (data[col] <= 0).sum()
            if negative_prices > 0:
                report['issues'].append(f'{col} 列发现 {negative_prices} 条非正价格记录')
                score -= 20
        
        # 检查价格连续性（避免异常跳跃）
        if len(data) > 1:
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in data.columns:
                    price_changes = data[col].pct_change().abs()
                    extreme_changes = (price_changes > 0.5).sum()  # 超过50%的变化
                    if extreme_changes > len(data) * 0.05:  # 超过5%的数据有极端变化
                        report['warnings'].append(f'{col} 列存在较多极端价格变化')
                        score -= 10
        
        return max(0, score)
    
    def _check_data_timeliness(self, data: pd.DataFrame, report: Dict[str, Any]) -> float:
        """检查数据时效性"""
        score = 100.0
        
        try:
            # 检查日期索引
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'Date' in data.columns:
                    data = data.set_index('Date')
                else:
                    report['warnings'].append('无法确定数据的日期信息')
                    return 70.0
            
            # 检查数据的时间范围
            if len(data) > 0:
                latest_date = data.index.max()
                oldest_date = data.index.min()
                
                # 检查数据是否过于陈旧
                days_since_latest = (datetime.now().date() - latest_date.date()).days
                if days_since_latest > 7:  # 超过一周
                    report['warnings'].append(f'数据可能不够新，最新数据距今 {days_since_latest} 天')
                    score -= min(20, days_since_latest)
                
                # 检查数据连续性
                expected_days = (latest_date - oldest_date).days + 1
                actual_days = len(data)
                completeness_ratio = actual_days / max(expected_days * 0.7, 1)  # 考虑周末和节假日
                
                if completeness_ratio < 0.8:
                    report['warnings'].append(f'数据连续性不足，完整度: {completeness_ratio:.1%}')
                    score -= 15
        
        except Exception as e:
            report['warnings'].append(f'时效性检查异常: {str(e)}')
            score = 70.0
        
        return max(0, score)
    
    def _detect_outliers(self, data: pd.DataFrame, report: Dict[str, Any]):
        """检测异常值"""
        try:
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_columns = [col for col in numeric_columns if col in data.columns]
            
            for col in available_columns:
                if col == 'Volume':
                    continue  # 成交量的异常值检测需要特殊处理
                
                # 使用IQR方法检测异常值
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                
                if len(outliers) > 0:
                    outlier_info = {
                        'column': col,
                        'count': len(outliers),
                        'percentage': len(outliers) / len(data) * 100,
                        'dates': outliers.index.strftime('%Y-%m-%d').tolist()[:5]  # 最多显示5个日期
                    }
                    report['outliers'].append(outlier_info)
                    
                    if outlier_info['percentage'] > 10:  # 超过10%为异常值
                        report['warnings'].append(
                            f'{col} 列异常值比例较高: {outlier_info["percentage"]:.1f}%'
                        )
        
        except Exception as e:
            report['warnings'].append(f'异常值检测失败: {str(e)}')
    
    def get_data_quality_report(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """获取指定股票的数据质量报告"""
        try:
            data = self.get_stock_data(symbol, start_date, end_date)
            return self._validate_data_detailed(data, symbol)
        except Exception as e:
            return {
                'is_valid': False,
                'symbol': symbol,
                'error': str(e),
                'quality_score': 0.0
            }
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str, timeout: int = 10) -> pd.DataFrame:
        """获取股票数据（主要接口）- 优化版本，支持缓存、熔断和并发"""
        # 1. 检查缓存
        cache = get_data_cache()
        cached_result = cache.get(symbol, start_date, end_date)
        if cached_result is not None:
            cached_data, source, age = cached_result
            logger.info(f"从缓存获取 {symbol} 数据，共 {len(cached_data)} 条记录，来源: {source}，缓存年龄: {age}秒")
            return cached_data
        
        market_type = self._detect_market_type(symbol)
        providers = self.providers.get(market_type, [])
        
        if not providers:
            raise ValueError(f"没有可用的数据源支持市场类型: {market_type}")
        
        # 2. 获取熔断器管理器并过滤可用数据源
        circuit_manager = get_circuit_breaker_manager()
        data_sources = self.config.get('data_sources', {})
        
        available_providers = []
        for provider in providers:
            # 检查数据源是否enabled
            source_name = provider.name.lower().replace(' ', '_')
            if 'yahoo' in source_name:
                enabled = data_sources.get('yahoo_finance', {}).get('enabled', True)
            elif '新浪' in provider.name or 'sina' in source_name:
                enabled = data_sources.get('sina_finance', {}).get('enabled', True)
            elif 'alpha' in source_name:
                enabled = data_sources.get('alpha_vantage', {}).get('enabled', False)
            elif 'tushare' in source_name:
                enabled = data_sources.get('tushare', {}).get('enabled', False)
            else:
                enabled = True  # 默认启用
            
            if not enabled:
                logger.debug(f"跳过disabled数据源: {provider.name}")
                continue
            
            # 检查熔断器状态
            circuit_breaker = circuit_manager.get_breaker(provider.name)
            if not circuit_breaker.can_execute():
                logger.debug(f"跳过熔断的数据源: {provider.name}")
                continue
            
            # 检查健康状态
            if self._should_perform_health_check(provider):
                self._perform_health_check(provider)
            
            # 跳过失败的数据源
            if provider.health.status == DataSourceStatus.FAILED:
                logger.debug(f"跳过失败的数据源: {provider.name}")
                continue
            
            available_providers.append(provider)
        
        if not available_providers:
            raise ValueError(f"没有可用的数据源支持市场类型: {market_type}")
        
        # 3. 使用并发获取器获取数据
        concurrent_fetcher = get_concurrent_fetcher()
        
        # 按优先级分离高低优先级数据源
        high_priority = [p for p in available_providers if p.priority <= 2]
        low_priority = [p for p in available_providers if p.priority > 2]
        
        try:
            # 使用混合策略获取数据
            result = concurrent_fetcher.fetch_data_hybrid(
                high_priority_providers=high_priority,
                low_priority_providers=low_priority,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if result and result.success and result.data is not None:
                # 验证数据质量
                if self._validate_data(result.data, symbol):
                    # 记录熔断器成功
                    circuit_breaker = circuit_manager.get_breaker(result.source)
                    circuit_breaker.record_success()
                    
                    # 更新提供者健康状态
                    for provider in available_providers:
                        if provider.name == result.source:
                            provider.update_health_success(result.response_time)
                            break
                    
                    # 缓存数据
                    cache.put(symbol, start_date, end_date, result.data, source=result.source)
                    
                    logger.info(f"成功从 {result.source} 获取 {symbol} 数据，共 {len(result.data)} 条记录，耗时 {result.response_time:.2f}秒")
                    return result.data
                else:
                    logger.warning(f"{result.source} 返回的数据质量不合格")
            
            # 如果并发获取失败，记录所有尝试过的数据源的熔断器失败
            for provider in available_providers:
                circuit_breaker = circuit_manager.get_breaker(provider.name)
                circuit_breaker.record_failure()
                provider.update_health_failure(Exception("并发获取失败"))
            
            raise Exception(f"所有数据源均无法获取 {symbol} 的数据: {result.error if result else '未知错误'}")
            
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            raise
    
    def get_realtime_data(self, symbol: str, timeout: int = 5) -> Dict[str, Any]:
        """获取实时数据 - 优化版本，支持缓存、熔断和并发"""
        # 1. 检查实时缓存
        cache = get_data_cache()
        cached_result = cache.get_realtime(symbol)
        if cached_result:
            data, source, age = cached_result
            logger.info(f"实时数据缓存命中: {symbol}, 来源: {source}, 年龄: {age}秒")
            return data
        
        market_type = self._detect_market_type(symbol)
        providers = self.providers.get(market_type, [])
        
        if not providers:
            raise ValueError(f"没有可用的数据源支持市场类型: {market_type}")
        
        # 2. 获取熔断器管理器并过滤可用数据源
        circuit_manager = get_circuit_breaker_manager()
        data_sources = self.config.get('data_sources', {})
        
        available_providers = []
        for provider in providers:
            # 检查数据源是否enabled
            source_name = provider.name.lower().replace(' ', '_')
            if 'yahoo' in source_name:
                enabled = data_sources.get('yahoo_finance', {}).get('enabled', True)
            elif '新浪' in provider.name or 'sina' in source_name:
                enabled = data_sources.get('sina_finance', {}).get('enabled', True)
            elif 'alpha' in source_name:
                enabled = data_sources.get('alpha_vantage', {}).get('enabled', False)
            elif 'tushare' in source_name:
                enabled = data_sources.get('tushare', {}).get('enabled', False)
            else:
                enabled = True  # 默认启用
            
            if not enabled:
                logger.debug(f"跳过disabled数据源: {provider.name}")
                continue
            
            # 检查熔断器状态
            circuit_breaker = circuit_manager.get_breaker(provider.name)
            if not circuit_breaker.can_execute():
                logger.debug(f"跳过熔断的数据源: {provider.name}")
                continue
            
            # 跳过失败的数据源
            if provider.health.status == DataSourceStatus.FAILED:
                logger.debug(f"跳过失败的数据源: {provider.name}")
                continue
            
            available_providers.append(provider)
        
        if not available_providers:
            raise ValueError(f"没有可用的数据源支持市场类型: {market_type}")
        
        # 3. 使用并发获取器获取实时数据
        concurrent_fetcher = get_concurrent_fetcher()
        
        # 按优先级分离高低优先级数据源
        high_priority = [p for p in available_providers if p.priority <= 2]
        low_priority = [p for p in available_providers if p.priority > 2]
        
        try:
            # 使用混合策略获取实时数据
            result = concurrent_fetcher.fetch_realtime_data_hybrid(
                high_priority_providers=high_priority,
                low_priority_providers=low_priority,
                symbol=symbol,
                timeout=timeout
            )
            
            if result and result.success and result.data is not None:
                # 记录熔断器成功
                circuit_breaker = circuit_manager.get_breaker(result.source)
                circuit_breaker.record_success()
                
                # 更新提供者健康状态
                for provider in available_providers:
                    if provider.name == result.source:
                        provider.update_health_success(result.response_time)
                        break
                
                # 缓存实时数据（较短的缓存时间）
                cache.put_realtime(symbol, result.data, source=result.source)
                
                logger.info(f"成功从 {result.source} 获取 {symbol} 实时数据，耗时 {result.response_time:.2f}秒")
                return result.data
            
            # 如果并发获取失败，记录所有尝试过的数据源的熔断器失败
            for provider in available_providers:
                circuit_breaker = circuit_manager.get_breaker(provider.name)
                circuit_breaker.record_failure()
                provider.update_health_failure(Exception("并发获取实时数据失败"))
            
            raise Exception(f"无法获取 {symbol} 的实时数据: {result.error if result else '未知错误'}")
            
        except Exception as e:
            logger.error(f"获取 {symbol} 实时数据失败: {e}")
            raise
    
    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有数据源的健康状态"""
        status = {}
        
        for market, providers in self.providers.items():
            status[market.value] = []
            for provider in providers:
                status[market.value].append({
                    'name': provider.name,
                    'status': provider.health.status.value,
                    'priority': provider.priority,
                    'last_success': provider.health.last_success,
                    'last_failure': provider.health.last_failure,
                    'failure_count': provider.health.failure_count,
                    'avg_response_time': provider.health.avg_response_time,
                    'success_rate': provider.health.success_rate
                })
        
        return status
    
    def force_health_check(self):
        """强制执行所有数据源的健康检查（仅检查enabled的数据源）"""
        logger.info("开始强制健康检查...")
        
        data_sources = self.config.get('data_sources', {})
        
        for market, providers in self.providers.items():
            for provider in providers:
                # 检查数据源是否enabled
                source_name = provider.name.lower().replace(' ', '_')
                if 'yahoo' in source_name:
                    enabled = data_sources.get('yahoo_finance', {}).get('enabled', True)
                elif '新浪' in provider.name or 'sina' in source_name:
                    enabled = data_sources.get('sina_finance', {}).get('enabled', True)
                elif 'alpha' in source_name:
                    enabled = data_sources.get('alpha_vantage', {}).get('enabled', False)
                elif 'tushare' in source_name:
                    enabled = data_sources.get('tushare', {}).get('enabled', False)
                else:
                    enabled = True  # 默认启用
                
                if not enabled:
                    logger.debug(f"跳过disabled数据源的健康检查: {provider.name}")
                    continue
                
                self._perform_health_check(provider)
        
        logger.info("健康检查完成")

# 全局实例
_global_manager: Optional[MultiSourceDataManager] = None

def get_multi_source_manager(config: Dict[str, Any] = None) -> MultiSourceDataManager:
    """获取全局多数据源管理器实例"""
    global _global_manager
    
    if _global_manager is None:
        # 如果没有提供配置，则从DataSourceConfig加载
        if config is None:
            try:
                from .data_source_config import DataSourceConfig
                config_manager = DataSourceConfig()
                config = config_manager.get_config()
                logger.info(f"已从DataSourceConfig加载配置: {list(config.get('data_sources', {}).keys())}")
            except Exception as e:
                logger.warning(f"无法加载DataSourceConfig，使用默认配置: {e}")
                config = {}
        
        _global_manager = MultiSourceDataManager(config)
    
    return _global_manager

def get_stock_data_with_fallback(symbol: str, start_date: str, end_date: str, config: Dict[str, Any] = None, timeout: int = 10) -> pd.DataFrame:
    """带故障转移的股票数据获取函数（便捷接口）- 优化版本"""
    manager = get_multi_source_manager(config)
    return manager.get_stock_data(symbol, start_date, end_date, timeout)

def get_realtime_data_with_fallback(symbol: str, config: Dict[str, Any] = None, timeout: int = 5) -> Dict[str, Any]:
    """带故障转移的实时数据获取函数（便捷接口）- 优化版本"""
    manager = get_multi_source_manager(config)
    return manager.get_realtime_data(symbol, timeout)