#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Vantage 数据工具模块

该模块提供 Alpha Vantage API 的数据获取功能，包括：
1. 股票价格数据
2. 技术指标数据
3. 基本面数据
4. 经济指标数据
"""

import requests
import pandas as pd
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from ..utils.logging_manager import get_logger
from ..utils.retry_utils import with_retry, safe_execute, get_error_message

logger = get_logger('dataflow', 'alpha_vantage')


class AlphaVantageAPI:
    """
    Alpha Vantage API 客户端
    """
    
    def __init__(self, api_key: str):
        """
        初始化 Alpha Vantage API 客户端
        
        Args:
            api_key: Alpha Vantage API 密钥
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
        
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        发送 API 请求
        
        Args:
            params: 请求参数
            
        Returns:
            Dict: API 响应数据
        """
        params['apikey'] = self.api_key
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 检查 API 限制
            if 'Note' in data:
                logger.warning(
                    "Alpha Vantage API rate limit reached",
                    extra={
                        'api_note': data['Note'],
                        'component': 'alpha_vantage',
                        'action': 'rate_limit_handling'
                    }
                )
                time.sleep(60)  # 等待1分钟后重试
                return self._make_request(params)
                
            # 检查错误信息
            if 'Error Message' in data:
                error_msg = data['Error Message']
                logger.error(
                    "Alpha Vantage API returned error",
                    extra={
                        'error_message': error_msg,
                        'params': params,
                        'component': 'alpha_vantage'
                    }
                )
                raise ValueError(f"Alpha Vantage API 错误: {error_msg}")
                
            logger.debug(
                "Alpha Vantage API request successful",
                extra={
                    'function': params.get('function'),
                    'symbol': params.get('symbol'),
                    'component': 'alpha_vantage'
                }
            )
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(
                "Alpha Vantage API request failed",
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'params': params,
                    'component': 'alpha_vantage'
                }
            )
            raise
    
    def get_daily_prices(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        """
        获取日线价格数据
        
        Args:
            symbol: 股票代码
            outputsize: 数据量 ("compact" 或 "full")
            
        Returns:
            DataFrame: 价格数据
        """
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize
        }
        
        data = self._make_request(params)
        
        if 'Time Series (Daily)' not in data:
            raise ValueError(f"无法获取 {symbol} 的价格数据")
            
        time_series = data['Time Series (Daily)']
        
        # 转换为 DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # 重命名列 (免费版本只有5列)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # 转换数据类型
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        return df
    
    def get_intraday_prices(self, symbol: str, interval: str = "5min") -> pd.DataFrame:
        """
        获取分钟级价格数据
        
        Args:
            symbol: 股票代码
            interval: 时间间隔 ("1min", "5min", "15min", "30min", "60min")
            
        Returns:
            DataFrame: 价格数据
        """
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval
        }
        
        data = self._make_request(params)
        
        time_series_key = f'Time Series ({interval})'
        if time_series_key not in data:
            raise ValueError(f"无法获取 {symbol} 的分钟数据")
            
        time_series = data[time_series_key]
        
        # 转换为 DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # 重命名列
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # 转换数据类型
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        return df
    
    def get_technical_indicator(self, symbol: str, indicator: str, **kwargs) -> pd.DataFrame:
        """
        获取技术指标数据
        
        Args:
            symbol: 股票代码
            indicator: 技术指标名称
            **kwargs: 指标参数
            
        Returns:
            DataFrame: 技术指标数据
        """
        params = {
            'function': indicator,
            'symbol': symbol,
            **kwargs
        }
        
        data = self._make_request(params)
        
        # 查找技术分析数据键
        tech_key = None
        for key in data.keys():
            if 'Technical Analysis' in key:
                tech_key = key
                break
                
        if not tech_key:
            raise ValueError(f"无法获取 {symbol} 的 {indicator} 数据")
            
        time_series = data[tech_key]
        
        # 转换为 DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # 转换数据类型
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        return df
    
    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        获取公司基本信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            Dict: 公司信息
        """
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol
        }
        
        return self._make_request(params)
    
    def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """
        获取财报数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            Dict: 财报数据
        """
        params = {
            'function': 'EARNINGS',
            'symbol': symbol
        }
        
        return self._make_request(params)
    
    def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        """
        获取损益表数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            Dict: 损益表数据
        """
        params = {
            'function': 'INCOME_STATEMENT',
            'symbol': symbol
        }
        
        return self._make_request(params)
    
    def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """
        获取资产负债表数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            Dict: 资产负债表数据
        """
        params = {
            'function': 'BALANCE_SHEET',
            'symbol': symbol
        }
        
        return self._make_request(params)
    
    def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """
        获取现金流量表数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            Dict: 现金流量表数据
        """
        params = {
            'function': 'CASH_FLOW',
            'symbol': symbol
        }
        
        return self._make_request(params)
    
    def get_economic_indicator(self, indicator: str, interval: str = "monthly") -> pd.DataFrame:
        """
        获取经济指标数据
        
        Args:
            indicator: 经济指标名称
            interval: 数据频率
            
        Returns:
            DataFrame: 经济指标数据
        """
        params = {
            'function': indicator,
            'interval': interval
        }
        
        data = self._make_request(params)
        
        # 查找数据键
        data_key = None
        for key in data.keys():
            if 'data' in key.lower():
                data_key = key
                break
                
        if not data_key:
            raise ValueError(f"无法获取经济指标 {indicator} 数据")
            
        time_series = data[data_key]
        
        # 转换为 DataFrame
        df = pd.DataFrame(time_series)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.sort_index()
            
        return df


def get_alpha_vantage_client() -> AlphaVantageAPI:
    """
    获取 Alpha Vantage API 客户端
    
    Returns:
        AlphaVantageAPI: API 客户端实例
    """
    from ..config_manager import get_config_manager
    
    config_manager = get_config_manager()
    api_key = config_manager.get_api_key('alpha_vantage')
    
    if not api_key:
        raise ValueError("Alpha Vantage API 密钥未配置")
        
    return AlphaVantageAPI(api_key)


# 便捷函数
@with_retry(max_retries=3, delay=2)
def get_stock_data_alpha_vantage(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取股票数据 (Alpha Vantage)
    
    Args:
        symbol: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        
    Returns:
        pd.DataFrame: 股票数据
    """
    logger.info(f"正在获取 Alpha Vantage 股票数据: {symbol} ({start_date} 到 {end_date})")
    
    def _fetch_stock_data():
        client = get_alpha_vantage_client()
        df = client.get_daily_prices(symbol, outputsize="full")
        
        # 过滤日期范围
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df_filtered = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        if df_filtered.empty:
            raise ValueError(f"未找到 {symbol} 在 {start_date} 到 {end_date} 期间的数据")
            
        logger.info(f"成功获取 Alpha Vantage 股票数据: {symbol}，数据行数: {len(df_filtered)}")
        return df_filtered
    
    # 使用安全执行函数
    result = safe_execute(_fetch_stock_data, fallback_value=None)
    
    if result is None:
        error_msg = f"无法获取 {symbol} 的 Alpha Vantage 股票数据，请稍后重试"
        logger.warning(error_msg)
        return pd.DataFrame()  # 返回空的 DataFrame
    
    return result


@with_retry(max_retries=3, delay=2)
def get_company_fundamentals_alpha_vantage(symbol: str) -> str:
    """获取公司基本面数据 (Alpha Vantage)
    
    Args:
        symbol: 股票代码
        
    Returns:
        str: 格式化的基本面数据
    """
    logger.info(f"正在获取 Alpha Vantage 基本面数据: {symbol}")
    
    def _fetch_fundamentals():
        client = get_alpha_vantage_client()
        
        # 获取公司概览
        overview = client.get_company_overview(symbol)
        
        # 获取财报数据
        earnings = client.get_earnings(symbol)
        
        # 格式化输出
        result = f"## {symbol} 基本面数据 (Alpha Vantage)\n\n"
        
        # 公司概览
        result += "### 公司概览\n"
        key_metrics = [
            'Name', 'Description', 'Sector', 'Industry', 'MarketCapitalization',
            'PERatio', 'PEGRatio', 'BookValue', 'DividendPerShare', 'DividendYield',
            'EPS', 'RevenuePerShareTTM', 'ProfitMargin', 'OperatingMarginTTM',
            'ReturnOnAssetsTTM', 'ReturnOnEquityTTM', 'RevenueTTM', 'GrossProfitTTM'
        ]
        
        for key in key_metrics:
            if key in overview and overview[key] != 'None':
                result += f"**{key}**: {overview[key]}\n"
        
        # 财报数据
        if 'quarterlyEarnings' in earnings:
            result += "\n### 季度财报\n"
            quarterly = earnings['quarterlyEarnings'][:4]  # 最近4个季度
            for quarter in quarterly:
                result += f"**{quarter['fiscalDateEnding']}**: EPS = {quarter['reportedEPS']}, 预期 EPS = {quarter['estimatedEPS']}\n"
        
        logger.info(f"成功获取 Alpha Vantage 基本面数据: {symbol}")
        return result
    
    # 使用安全执行函数
    result = safe_execute(_fetch_fundamentals, fallback_value=None)
    
    if result is None:
        error_msg = f"无法获取 {symbol} 的 Alpha Vantage 基本面数据，请稍后重试"
        logger.warning(error_msg)
        return f"数据获取失败: {error_msg}"
    
    return result


@with_retry(max_retries=3, delay=2)
def get_technical_indicators_alpha_vantage(symbol: str, indicator: str, **kwargs) -> str:
    """获取技术指标数据 (Alpha Vantage)
    
    Args:
        symbol: 股票代码
        indicator: 技术指标名称
        **kwargs: 指标参数
        
    Returns:
        str: 格式化的技术指标数据
    """
    logger.info(f"正在获取 Alpha Vantage 技术指标: {symbol} {indicator}")
    
    def _fetch_indicators():
        client = get_alpha_vantage_client()
        df = client.get_technical_indicator(symbol, indicator, **kwargs)
        
        # 获取最近20个数据点
        df_recent = df.tail(20)
        
        if df_recent.empty:
            raise ValueError(f"未找到 {symbol} 的 {indicator} 技术指标数据")
        
        result = f"## {symbol} {indicator} 技术指标 (Alpha Vantage)\n\n"
        result += df_recent.to_string()
        
        logger.info(f"成功获取 Alpha Vantage 技术指标: {symbol} {indicator}，数据行数: {len(df_recent)}")
        return result
    
    # 使用安全执行函数
    result = safe_execute(_fetch_indicators, fallback_value=None)
    
    if result is None:
        error_msg = f"无法获取 {symbol} 的 {indicator} 技术指标数据，请稍后重试"
        logger.warning(error_msg)
        return f"数据获取失败: {error_msg}"
    
    return result