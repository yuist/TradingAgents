import pandas as pd
import yfinance as yf
from stockstats import wrap
from typing import Annotated, Dict, Any, Optional
import os
from datetime import datetime
import requests.exceptions
import socket
import hashlib
import json
import time
from functools import lru_cache
from ..utils.logging_manager import get_logger


class StockstatsUtils:
    # 类级别的缓存字典
    _indicator_cache: Dict[str, Dict[str, Any]] = {}
    _data_cache: Dict[str, pd.DataFrame] = {}
    _cache_ttl = 3600  # 缓存1小时
    _cache_max_size = 1000  # 最大缓存条目数
    
    @classmethod
    def _get_cache_key(cls, symbol: str, indicator: str, curr_date: str) -> str:
        """生成缓存键"""
        key_data = f"{symbol}_{indicator}_{curr_date}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    @classmethod
    def _is_cache_valid(cls, cache_entry: Dict[str, Any]) -> bool:
        """检查缓存是否有效"""
        if 'timestamp' not in cache_entry:
            return False
        return time.time() - cache_entry['timestamp'] < cls._cache_ttl
    
    @classmethod
    def _get_from_cache(cls, cache_key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if cache_key in cls._indicator_cache:
            cache_entry = cls._indicator_cache[cache_key]
            if cls._is_cache_valid(cache_entry):
                logger = get_logger('dataflow', 'stockstats')
                logger.info(
                    "Cache hit",
                    extra={
                        'cache_key_prefix': cache_key[:8],
                        'component': 'stockstats_utils',
                        'action': 'cache_hit'
                    }
                )
                return cache_entry['value']
            else:
                # 清理过期缓存
                del cls._indicator_cache[cache_key]
                logger = get_logger('dataflow', 'stockstats')
                logger.info(
                    "Cache expired",
                    extra={
                        'cache_key_prefix': cache_key[:8],
                        'component': 'stockstats_utils',
                        'action': 'cache_expired'
                    }
                )
        return None
    
    @classmethod
    def _set_cache(cls, cache_key: str, value: Any) -> None:
        """设置缓存"""
        cls._indicator_cache[cache_key] = {
            'value': value,
            'timestamp': time.time()
        }
        logger = get_logger('dataflow', 'stockstats')
        logger.info(
            "Result cached",
            extra={
                'cache_key_prefix': cache_key[:8],
                'component': 'stockstats_utils',
                'action': 'cache_set'
            }
        )
        
        # 限制缓存大小，防止内存溢出
        if len(cls._indicator_cache) > cls._cache_max_size:
            # 删除最旧的缓存项
            oldest_key = min(cls._indicator_cache.keys(), 
                           key=lambda k: cls._indicator_cache[k]['timestamp'])
            del cls._indicator_cache[oldest_key]
            logger = get_logger('dataflow', 'stockstats')
            logger.info(
                "Cache size limit reached, removed oldest entry",
                extra={
                    'component': 'stockstats_utils',
                    'cache_max_size': cls._cache_max_size,
                    'action': 'cache_cleanup'
                }
            )
    
    @classmethod
    def clear_cache(cls) -> None:
        """清理所有缓存"""
        cls._indicator_cache.clear()
        cls._data_cache.clear()
        logger = get_logger('dataflow', 'stockstats')
        logger.info(
            "All caches cleared",
            extra={'component': 'stockstats_utils', 'action': 'cache_clear'}
        )
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        """获取缓存统计信息"""
        valid_entries = sum(1 for entry in cls._indicator_cache.values() 
                          if cls._is_cache_valid(entry))
        return {
            'total_entries': len(cls._indicator_cache),
            'valid_entries': valid_entries,
            'data_cache_entries': len(cls._data_cache)
        }
    @staticmethod
    def _validate_inputs(symbol: str, indicator: str, curr_date: str, data_dir: str) -> None:
        """验证输入参数的有效性"""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        if not indicator or not isinstance(indicator, str):
            raise ValueError("Indicator must be a non-empty string")
        
        if not curr_date or not isinstance(curr_date, str):
            raise ValueError("Current date must be a non-empty string")
        
        # 验证日期格式
        try:
            datetime.strptime(curr_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Current date must be in YYYY-mm-dd format")
        
        if not data_dir or not isinstance(data_dir, str):
            raise ValueError("Data directory must be a non-empty string")
    
    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        """验证DataFrame的有效性"""
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(df) < 20:  # 大多数技术指标需要至少20个数据点
            logger = get_logger('dataflow', 'stockstats')
            logger.warning(
                "DataFrame has insufficient data for accurate indicators",
                extra={
                    'component': 'stockstats_utils',
                    'row_count': len(df),
                    'minimum_required': 20,
                    'function': '_validate_dataframe'
                }
            )
    
    @classmethod
    def get_stock_stats(
        cls,
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
        data_dir: Annotated[
            str,
            "directory where the stock data is stored.",
        ],
        online: Annotated[
            bool,
            "whether to use online tools to fetch data or offline tools. If True, will use online tools.",
        ] = False,
        use_cache: Annotated[
            bool,
            "whether to use caching for performance optimization",
        ] = True,
    ):
        # 输入验证
        try:
            cls._validate_inputs(symbol, indicator, curr_date, data_dir)
        except ValueError as e:
            logger = get_logger('dataflow', 'stockstats')
            logger.error(
                "Input validation failed",
                extra={
                    'component': 'stockstats_utils',
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'function': 'get_stock_stats'
                }
            )
            return f"Error: {e}"
        
        # 检查缓存
        if use_cache:
            cache_key = cls._get_cache_key(symbol, indicator, curr_date)
            cached_result = cls._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
        
        df = None
        data = None

        if not online:
            # 离线模式：从本地文件读取数据
            try:
                file_path = os.path.join(
                    data_dir,
                    f"{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
                )
                
                if not os.path.exists(file_path):
                    logger = get_logger('dataflow', 'stockstats')
                    logger.error(
                        "Data file not found",
                        extra={
                            'component': 'stockstats_utils',
                            'file_path': file_path,
                            'symbol': symbol,
                            'function': 'get_stock_stats'
                        }
                    )
                    return "Error: Stock data file not found. Please ensure data is downloaded first."
                
                if not os.access(file_path, os.R_OK):
                    logger = get_logger('dataflow', 'stockstats')
                    logger.error(
                        "No read permission for data file",
                        extra={
                            'component': 'stockstats_utils',
                            'file_path': file_path,
                            'symbol': symbol,
                            'function': 'get_stock_stats'
                        }
                    )
                    return "Error: No permission to read data file."
                
                data = pd.read_csv(file_path)
                logger = get_logger('dataflow', 'stockstats')
                logger.info(
                    "Successfully loaded offline data",
                    extra={
                        'component': 'stockstats_utils',
                        'symbol': symbol,
                        'file_path': file_path,
                        'data_rows': len(data),
                        'function': 'get_stock_stats'
                    }
                )
                
                # 确保 Date 列是 datetime 类型
                data["Date"] = pd.to_datetime(data["Date"])
                
                # 验证数据完整性
                cls._validate_dataframe(data)
                
                df = wrap(data)
                
            except FileNotFoundError:
                logger = get_logger('dataflow', 'stockstats')
                logger.error(f"Stock data file not found for symbol {symbol}")
                return "Error: Yahoo Finance data not fetched yet! Please download data first."
            except pd.errors.EmptyDataError:
                logger = get_logger('dataflow', 'stockstats')
                logger.error(f"Data file is empty for symbol {symbol}")
                return "Error: Stock data file is empty."
            except pd.errors.ParserError as e:
                logger = get_logger('dataflow', 'stockstats')
                logger.error(f"Failed to parse CSV file for symbol {symbol}: {e}")
                return "Error: Failed to parse stock data file."
            except Exception as e:
                logger = get_logger('dataflow', 'stockstats')
                logger.error(f"Unexpected error loading offline data for {symbol}: {e}")
                return f"Error: Failed to load stock data - {str(e)}"
        else:
            # 在线模式：使用多数据源管理器获取数据
            try:
                # Get today's date as YYYY-mm-dd to add to cache
                today_date = pd.Timestamp.today()
                curr_date_dt = pd.to_datetime(curr_date)

                end_date = today_date
                start_date = today_date - pd.DateOffset(years=5)
                start_date = start_date.strftime("%Y-%m-%d")
                end_date = end_date.strftime("%Y-%m-%d")

                # Get config and ensure cache directory exists
                try:
                    from ..config_manager import get_config_manager
                    config_manager = get_config_manager()
                    config = config_manager.config
                    
                    # 获取数据目录配置
                    data_cache_dir = config["system"].get("data_dir", "./data")
                    if not os.path.isabs(data_cache_dir):
                        data_cache_dir = os.path.join(config["system"]["project_dir"], data_cache_dir.lstrip("./"))
                except Exception as e:
                    logger = get_logger('dataflow', 'stockstats')
                    logger.error(f"Failed to get config: {e}")
                    data_cache_dir = "./data"  # 使用默认目录
                
                try:
                    os.makedirs(data_cache_dir, exist_ok=True)
                except OSError as e:
                    logger = get_logger('dataflow', 'stockstats')
                    logger.error(f"Failed to create cache directory {data_cache_dir}: {e}")
                    return "Error: Failed to create data cache directory."

                # 使用多数据源管理器获取数据
                try:
                    from .multi_source_manager import MultiSourceDataManager
                    from .data_source_config import DataSourceConfig
                    
                    # 初始化数据源配置和管理器
                    config_manager = DataSourceConfig()
                    data_manager = MultiSourceDataManager(config_manager.config)
                    
                    logger = get_logger('dataflow', 'stockstats')
                    logger.info(f"Using multi-source data manager to get data for {symbol} from {start_date} to {end_date}")
                    data = data_manager.get_stock_data(symbol, start_date, end_date)
                    
                    if data is None or data.empty:
                        logger.error(f"No data returned for symbol {symbol} from multi-source manager")
                        return "Error: No data available for this symbol. Please check if the symbol is valid."
                    
                    # 确保数据格式正确
                    if 'Date' not in data.columns and data.index.name == 'Date':
                        data = data.reset_index()
                    elif 'Date' not in data.columns:
                        # 如果没有Date列，使用索引作为Date
                        data['Date'] = data.index
                        data = data.reset_index(drop=True)
                    
                    logger.info(f"Successfully retrieved data for {symbol} using multi-source manager, {len(data)} records")
                    
                except ImportError as e:
                    logger.warning(f"Multi-source manager not available, falling back to Yahoo Finance: {e}")
                    # 回退到原来的Yahoo Finance方法
                    normalized_symbol = symbol.replace('.sz', '.SZ').replace('.sh', '.SS')
                    
                    logger.info(f"Downloading data for {normalized_symbol} (original: {symbol}) from {start_date} to {end_date}")
                    data = yf.download(
                        normalized_symbol,
                        start=start_date,
                        end=end_date,
                        multi_level_index=False,
                        progress=False,
                        auto_adjust=True,
                        timeout=30,  # 添加30秒超时
                    )
                    
                    if data.empty:
                        logger.error(f"No data returned for symbol {symbol}")
                        return "Error: No data available for this symbol. Please check if the symbol is valid."
                    
                    data = data.reset_index()
                    
                except Exception as e:
                    logger.error(f"Multi-source manager failed, falling back to Yahoo Finance: {e}")
                    # 回退到原来的Yahoo Finance方法
                    normalized_symbol = symbol.replace('.sz', '.SZ').replace('.sh', '.SS')
                    
                    try:
                        logger.info(f"Downloading data for {normalized_symbol} (original: {symbol}) from {start_date} to {end_date}")
                        data = yf.download(
                            normalized_symbol,
                            start=start_date,
                            end=end_date,
                            multi_level_index=False,
                            progress=False,
                            auto_adjust=True,
                            timeout=30,  # 添加30秒超时
                        )
                        
                        if data.empty:
                            logger.error(f"No data returned for symbol {symbol}")
                            return "Error: No data available for this symbol. Please check if the symbol is valid."
                        
                        data = data.reset_index()
                        
                    except (requests.exceptions.RequestException, socket.error) as e:
                        logger.error(f"Network error downloading data for {symbol}: {e}")
                        return "Error: Network connection failed. Please check your internet connection."
                    except Exception as e:
                        logger.error(f"Failed to download data for {symbol}: {e}")
                        return f"Error: Failed to download stock data - {str(e)}"
                
                # 验证数据完整性
                cls._validate_dataframe(data)
                
                df = wrap(data)
                
            except Exception as e:
                logger.error(f"Unexpected error in online mode for {symbol}: {e}")
                return f"Error: Failed to process stock data - {str(e)}"
        
        # 技术指标计算和日期匹配
        try:
            # 统一处理日期格式
            curr_date = pd.to_datetime(curr_date).strftime("%Y-%m-%d")
            
            # 确保 Date 列是字符串格式用于匹配
            if df["Date"].dtype == 'datetime64[ns]':
                df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            elif df["Date"].dtype == 'object':
                # 如果已经是字符串，确保格式正确
                try:
                    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
                except Exception as e:
                    logger = get_logger('dataflow', 'stockstats')
                    logger.error(f"Failed to convert Date column to proper format: {e}")
                    return "Error: Invalid date format in data."

            # 触发stockstats计算指标
            try:
                # 检查指标是否已经存在于DataFrame中
                if indicator in df.columns:
                    logger.info(f"Indicator {indicator} already exists in DataFrame")
                else:
                    # 尝试计算指标
                    calculated_column = df[indicator]  # trigger stockstats to calculate the indicator
                    if calculated_column is not None:
                        logger.info(f"Successfully calculated indicator {indicator} for {symbol}")
                    else:
                        logger.error(f"Indicator {indicator} calculation returned None")
                        return f"Error: Failed to calculate indicator '{indicator}' - calculation returned None"
            except KeyError:
                logger = get_logger('dataflow', 'stockstats')
                logger.error(f"Invalid indicator: {indicator}")
                return f"Error: Invalid indicator '{indicator}'. Please check the indicator name."
            except AttributeError as e:
                logger = get_logger('dataflow', 'stockstats')
                logger.error(f"StockStats attribute error for indicator {indicator}: {e}")
                return f"Error: Invalid indicator '{indicator}' or insufficient data for calculation."
            except TypeError as e:
                logger = get_logger('dataflow', 'stockstats')
                logger.error(f"Type error calculating indicator {indicator}: {e}")
                return f"Error: Data type issue calculating indicator '{indicator}'. Please check data format."
            except Exception as e:
                logger = get_logger('dataflow', 'stockstats')
                logger.error(f"Failed to calculate indicator {indicator}: {e}")
                return f"Error: Failed to calculate indicator '{indicator}' - {str(e)}"
            
            # 验证指标列是否存在
            if indicator not in df.columns:
                logger = get_logger('dataflow', 'stockstats')
                logger.error(f"Indicator {indicator} not found in DataFrame columns")
                return f"Error: Indicator '{indicator}' was not calculated successfully."
            
            # 首先尝试精确匹配
            try:
                matching_rows = df[df["Date"].str.startswith(curr_date)]
            except Exception as e:
                logger = get_logger('dataflow', 'stockstats')
                logger.error(f"Error during date matching: {e}")
                return "Error: Failed to match date in data."

            if not matching_rows.empty:
                 try:
                     indicator_value = matching_rows[indicator].values[0]
                     # 检查是否为有效数值
                     if pd.isna(indicator_value):
                         result = f"N/A: Indicator '{indicator}' not available for {curr_date} (insufficient data)"
                         logger = get_logger('dataflow', 'stockstats')
                         logger.warning(f"Indicator {indicator} returned NaN for date {curr_date}")
                     else:
                         result = indicator_value
                     
                     # 缓存结果
                     if use_cache:
                         cls._set_cache(cache_key, result)
                     
                     return result
                 except Exception as e:
                     logger = get_logger('dataflow', 'stockstats')
                     logger.error(f"Error extracting indicator value: {e}")
                     return "Error: Failed to extract indicator value."
            else:
                # 如果没有精确匹配，查找最近的交易日数据
                try:
                    # 将Date列转换为datetime进行比较
                    df_temp = df.copy()
                    df_temp["Date"] = pd.to_datetime(df_temp["Date"])
                    curr_date_dt = pd.to_datetime(curr_date)
                    
                    # 查找小于等于查询日期的最近交易日
                    available_dates = df_temp[df_temp["Date"] <= curr_date_dt]
                    
                    if not available_dates.empty:
                        # 获取最近的交易日
                        latest_date = available_dates["Date"].max()
                        latest_row = df_temp[df_temp["Date"] == latest_date]
                        
                        if not latest_row.empty:
                            indicator_value = latest_row[indicator].values[0]
                            # 检查是否为有效数值
                            if pd.isna(indicator_value):
                                result = f"N/A: Indicator '{indicator}' not available (insufficient data)"
                                logger = get_logger('dataflow', 'stockstats')
                                logger.warning(f"Indicator {indicator} returned NaN for fallback date")
                            else:
                                latest_date_str = latest_date.strftime("%Y-%m-%d")
                                result = f"{indicator_value} (from {latest_date_str})"
                            
                            # 缓存结果
                            if use_cache:
                                cls._set_cache(cache_key, result)
                            
                            return result
                    
                    return "N/A: No trading data available"
                    
                except Exception as e:
                    logger = get_logger('dataflow', 'stockstats')
                    logger.error(f"Error during date fallback logic: {e}")
                    return "Error: Failed to find fallback trading date."
                    
        except Exception as e:
            logger = get_logger('dataflow', 'stockstats')
            logger.error(f"Unexpected error during indicator calculation for {symbol}: {e}")
            return f"Error: Unexpected error - {str(e)}"
