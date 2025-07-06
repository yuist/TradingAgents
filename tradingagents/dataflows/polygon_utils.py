#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polygon.io 数据工具模块

该模块提供 Polygon.io API 的数据获取功能，包括：
1. 股票价格数据
2. 期权数据
3. 外汇数据
4. 加密货币数据
5. 市场新闻和情绪数据
"""

import requests
import pandas as pd
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from ..utils.retry_utils import with_retry, safe_execute, get_error_message
from ..utils.logging_manager import get_logger

logger = get_logger('dataflow', 'polygon')


class PolygonAPI:
    """
    Polygon.io API 客户端
    """
    
    def __init__(self, api_key: str):
        """
        初始化 Polygon API 客户端
        
        Args:
            api_key: Polygon API 密钥
        """
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        发送 API 请求
        
        Args:
            endpoint: API 端点
            params: 请求参数
            
        Returns:
            Dict: API 响应数据
        """
        if params is None:
            params = {}
            
        params['apikey'] = self.api_key
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 检查 API 状态
            if data.get('status') == 'ERROR':
                raise ValueError(f"Polygon API 错误: {data.get('error', '未知错误')}")
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(
                "Polygon API request failed",
                extra={
                    'component': 'polygon_utils',
                    'endpoint': endpoint,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'function': '_make_request'
                }
            )
            raise
    
    def get_aggregates(self, ticker: str, multiplier: int, timespan: str, 
                      from_date: str, to_date: str, adjusted: bool = True) -> pd.DataFrame:
        """
        获取聚合数据 (OHLCV)
        
        Args:
            ticker: 股票代码
            multiplier: 时间倍数
            timespan: 时间跨度 ("minute", "hour", "day", "week", "month", "quarter", "year")
            from_date: 开始日期 (YYYY-MM-DD)
            to_date: 结束日期 (YYYY-MM-DD)
            adjusted: 是否调整价格
            
        Returns:
            DataFrame: 聚合数据
        """
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            'adjusted': str(adjusted).lower(),
            'sort': 'asc',
            'limit': 50000
        }
        
        data = self._make_request(endpoint, params)
        
        if 'results' not in data or not data['results']:
            raise ValueError(f"无法获取 {ticker} 的聚合数据")
            
        results = data['results']
        
        # 转换为 DataFrame
        df = pd.DataFrame(results)
        
        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('timestamp')
        
        # 重命名列
        column_mapping = {
            'o': 'Open',
            'h': 'High', 
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume',
            'vw': 'VWAP',
            'n': 'Transactions'
        }
        
        df = df.rename(columns=column_mapping)
        
        # 选择需要的列
        columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'VWAP' in df.columns:
            columns_to_keep.append('VWAP')
        if 'Transactions' in df.columns:
            columns_to_keep.append('Transactions')
            
        df = df[columns_to_keep]
        
        return df
    
    def get_daily_open_close(self, ticker: str, date: str, adjusted: bool = True) -> Dict[str, Any]:
        """
        获取特定日期的开盘收盘价
        
        Args:
            ticker: 股票代码
            date: 日期 (YYYY-MM-DD)
            adjusted: 是否调整价格
            
        Returns:
            Dict: 开盘收盘数据
        """
        endpoint = f"/v1/open-close/{ticker}/{date}"
        params = {'adjusted': str(adjusted).lower()}
        
        return self._make_request(endpoint, params)
    
    def get_previous_close(self, ticker: str, adjusted: bool = True) -> Dict[str, Any]:
        """
        获取前一交易日收盘价
        
        Args:
            ticker: 股票代码
            adjusted: 是否调整价格
            
        Returns:
            Dict: 前一交易日数据
        """
        endpoint = f"/v2/aggs/ticker/{ticker}/prev"
        params = {'adjusted': str(adjusted).lower()}
        
        return self._make_request(endpoint, params)
    
    def get_ticker_details(self, ticker: str) -> Dict[str, Any]:
        """
        获取股票详细信息
        
        Args:
            ticker: 股票代码
            
        Returns:
            Dict: 股票详细信息
        """
        endpoint = f"/v3/reference/tickers/{ticker}"
        
        return self._make_request(endpoint)
    
    def get_ticker_news(self, ticker: Optional[str] = None, published_utc_gte: Optional[str] = None,
                       published_utc_lte: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取股票新闻
        
        Args:
            ticker: 股票代码 (可选)
            published_utc_gte: 发布时间下限 (YYYY-MM-DD)
            published_utc_lte: 发布时间上限 (YYYY-MM-DD)
            limit: 返回数量限制
            
        Returns:
            List: 新闻列表
        """
        endpoint = "/v2/reference/news"
        params = {'limit': limit}
        
        if ticker:
            params['ticker'] = ticker
        if published_utc_gte:
            params['published_utc.gte'] = published_utc_gte
        if published_utc_lte:
            params['published_utc.lte'] = published_utc_lte
            
        data = self._make_request(endpoint, params)
        
        return data.get('results', [])
    
    def get_market_status(self) -> Dict[str, Any]:
        """
        获取市场状态
        
        Returns:
            Dict: 市场状态信息
        """
        endpoint = "/v1/marketstatus/now"
        
        return self._make_request(endpoint)
    
    def get_market_holidays(self) -> List[Dict[str, Any]]:
        """
        获取市场假期
        
        Returns:
            List: 市场假期列表
        """
        endpoint = "/v1/marketstatus/upcoming"
        
        data = self._make_request(endpoint)
        return data.get('results', [])
    
    def get_financials(self, ticker: str, reporting_period: Optional[str] = None,
                      timeframe: str = "annual", limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取财务数据
        
        Args:
            ticker: 股票代码
            reporting_period: 报告期 (YYYY-MM-DD)
            timeframe: 时间框架 ("annual" 或 "quarterly")
            limit: 返回数量限制
            
        Returns:
            List: 财务数据列表
        """
        endpoint = f"/vX/reference/financials"
        params = {
            'ticker': ticker,
            'timeframe': timeframe,
            'limit': limit
        }
        
        if reporting_period:
            params['reporting_period'] = reporting_period
            
        data = self._make_request(endpoint, params)
        return data.get('results', [])
    
    def get_dividends(self, ticker: str, ex_dividend_date_gte: Optional[str] = None,
                     ex_dividend_date_lte: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取股息数据
        
        Args:
            ticker: 股票代码
            ex_dividend_date_gte: 除息日下限 (YYYY-MM-DD)
            ex_dividend_date_lte: 除息日上限 (YYYY-MM-DD)
            limit: 返回数量限制
            
        Returns:
            List: 股息数据列表
        """
        endpoint = f"/v3/reference/dividends"
        params = {
            'ticker': ticker,
            'limit': limit
        }
        
        if ex_dividend_date_gte:
            params['ex_dividend_date.gte'] = ex_dividend_date_gte
        if ex_dividend_date_lte:
            params['ex_dividend_date.lte'] = ex_dividend_date_lte
            
        data = self._make_request(endpoint, params)
        return data.get('results', [])
    
    def get_splits(self, ticker: str, execution_date_gte: Optional[str] = None,
                  execution_date_lte: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取股票分割数据
        
        Args:
            ticker: 股票代码
            execution_date_gte: 执行日期下限 (YYYY-MM-DD)
            execution_date_lte: 执行日期上限 (YYYY-MM-DD)
            limit: 返回数量限制
            
        Returns:
            List: 股票分割数据列表
        """
        endpoint = f"/v3/reference/splits"
        params = {
            'ticker': ticker,
            'limit': limit
        }
        
        if execution_date_gte:
            params['execution_date.gte'] = execution_date_gte
        if execution_date_lte:
            params['execution_date.lte'] = execution_date_lte
            
        data = self._make_request(endpoint, params)
        return data.get('results', [])


def get_polygon_client() -> PolygonAPI:
    """
    获取 Polygon API 客户端
    
    Returns:
        PolygonAPI: API 客户端实例
    """
    from ..config_manager import get_config_manager
    
    config_manager = get_config_manager()
    api_key = config_manager.get_api_key('polygon')
    
    if not api_key:
        raise ValueError("Polygon API 密钥未配置")
        
    return PolygonAPI(api_key)


# 便捷函数
@with_retry(max_retries=3, delay=2)
def get_stock_data_polygon(symbol: str, start_date: str, end_date: str) -> str:
    """获取股票数据 (Polygon)
    
    Args:
        symbol: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        
    Returns:
        str: 格式化的股票数据
    """
    logger.info(f"正在获取 Polygon 股票数据: {symbol} ({start_date} 到 {end_date})")
    
    def _fetch_stock_data():
        client = get_polygon_client()
        df = client.get_aggregates(symbol, 1, "day", start_date, end_date)
        
        if df.empty:
            raise ValueError(f"未找到 {symbol} 在 {start_date} 到 {end_date} 期间的数据")
            
        # 格式化输出
        result = f"## {symbol} 股票数据 (Polygon) - {start_date} 到 {end_date}\n\n"
        result += df.to_string()
        
        logger.info(f"成功获取 Polygon 股票数据: {symbol}，数据行数: {len(df)}")
        return result
    
    # 使用安全执行函数
    result = safe_execute(_fetch_stock_data, fallback_value=None)
    
    if result is None:
        error_msg = f"无法获取 {symbol} 的 Polygon 股票数据，请稍后重试"
        logger.warning(error_msg)
        return f"数据获取失败: {error_msg}"
    
    return result


@with_retry(max_retries=3, delay=2)
def get_company_news_polygon(symbol: str, start_date: str, end_date: str, limit: int = 10) -> str:
    """
    获取公司新闻 (Polygon)
    
    Args:
        symbol: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        limit: 新闻数量限制
        
    Returns:
        str: 格式化的新闻数据
    """
    logger.info(f"正在获取 Polygon 公司新闻: {symbol} ({start_date} 到 {end_date})")
    
    def _fetch_company_news():
        client = get_polygon_client()
        news_list = client.get_ticker_news(
            ticker=symbol,
            published_utc_gte=start_date,
            published_utc_lte=end_date,
            limit=limit
        )
        
        if not news_list:
            raise ValueError(f"未找到 {symbol} 在 {start_date} 到 {end_date} 期间的新闻")
            
        # 格式化输出
        result = f"## {symbol} 新闻 (Polygon) - {start_date} 到 {end_date}\n\n"
        
        for news in news_list:
            title = news.get('title', '无标题')
            description = news.get('description', '无描述')
            published_utc = news.get('published_utc', '未知时间')
            author = news.get('author', '未知作者')
            
            result += f"### {title}\n"
            result += f"**发布时间**: {published_utc}\n"
            result += f"**作者**: {author}\n"
            result += f"**描述**: {description}\n\n"
        
        logger.info(f"成功获取 Polygon 公司新闻: {symbol}，新闻数量: {len(news_list)}")
        return result
    
    # 使用安全执行函数
    result = safe_execute(_fetch_company_news, fallback_value=None)
    
    if result is None:
        error_msg = f"无法获取 {symbol} 的 Polygon 公司新闻，请稍后重试"
        logger.warning(error_msg)
        return f"数据获取失败: {error_msg}"
    
    return result


@with_retry(max_retries=3, delay=2)
def get_company_financials_polygon(symbol: str, timeframe: str = "annual") -> str:
    """
    获取公司财务数据 (Polygon)
    
    Args:
        symbol: 股票代码
        timeframe: 时间框架 ("annual" 或 "quarterly")
        
    Returns:
        str: 格式化的财务数据
    """
    logger.info(f"正在获取 Polygon 财务数据: {symbol} ({timeframe})")
    
    def _fetch_company_financials():
        client = get_polygon_client()
        
        # 获取公司详细信息
        details = client.get_ticker_details(symbol)
        
        # 获取财务数据
        financials = client.get_financials(symbol, timeframe=timeframe, limit=4)
        
        # 格式化输出
        result = f"## {symbol} 财务数据 (Polygon) - {timeframe}\n\n"
        
        # 公司基本信息
        if 'results' in details:
            company_info = details['results']
            result += "### 公司信息\n"
            result += f"**名称**: {company_info.get('name', 'N/A')}\n"
            result += f"**描述**: {company_info.get('description', 'N/A')}\n"
            result += f"**行业**: {company_info.get('sic_description', 'N/A')}\n"
            result += f"**市值**: {company_info.get('market_cap', 'N/A')}\n"
            result += f"**员工数**: {company_info.get('total_employees', 'N/A')}\n\n"
        
        # 财务数据
        if financials:
            result += "### 财务数据\n"
            for financial in financials[:2]:  # 显示最近2期
                period = financial.get('end_date', '未知期间')
                result += f"\n**报告期**: {period}\n"
                
                # 提取关键财务指标
                if 'financials' in financial:
                    fin_data = financial['financials']
                    
                    # 损益表数据
                    if 'income_statement' in fin_data:
                        income = fin_data['income_statement']
                        result += f"**营收**: {income.get('revenues', {}).get('value', 'N/A')}\n"
                        result += f"**净利润**: {income.get('net_income_loss', {}).get('value', 'N/A')}\n"
                        result += f"**每股收益**: {income.get('basic_earnings_per_share', {}).get('value', 'N/A')}\n"
                    
                    # 资产负债表数据
                    if 'balance_sheet' in fin_data:
                        balance = fin_data['balance_sheet']
                        result += f"**总资产**: {balance.get('assets', {}).get('value', 'N/A')}\n"
                        result += f"**总负债**: {balance.get('liabilities', {}).get('value', 'N/A')}\n"
                        result += f"**股东权益**: {balance.get('equity', {}).get('value', 'N/A')}\n"
                    
                    # 现金流量表数据
                    if 'cash_flow_statement' in fin_data:
                        cash_flow = fin_data['cash_flow_statement']
                        result += f"**经营现金流**: {cash_flow.get('net_cash_flow_from_operating_activities', {}).get('value', 'N/A')}\n"
        
        logger.info(f"成功获取 Polygon 财务数据: {symbol}")
        return result
    
    # 使用安全执行函数
    result = safe_execute(_fetch_company_financials, fallback_value=None)
    
    if result is None:
        error_msg = f"无法获取 {symbol} 的 Polygon 财务数据，请稍后重试"
        logger.warning(error_msg)
        return f"数据获取失败: {error_msg}"
    
    return result


@with_retry(max_retries=3, delay=2)
def get_market_status_polygon() -> str:
    """
    获取市场状态 (Polygon)
    
    Returns:
        str: 格式化的市场状态
    """
    logger.info("正在获取 Polygon 市场状态")
    
    def _fetch_market_status():
        client = get_polygon_client()
        status = client.get_market_status()
        
        result = "## 市场状态 (Polygon)\n\n"
        
        if 'market' in status:
            result += f"**市场状态**: {status['market']}\n"
        if 'serverTime' in status:
            result += f"**服务器时间**: {status['serverTime']}\n"
        if 'exchanges' in status:
            result += "\n### 交易所状态\n"
            for exchange, info in status['exchanges'].items():
                result += f"**{exchange}**: {info}\n"
        
        logger.info("成功获取 Polygon 市场状态")
        return result
    
    # 使用安全执行函数
    result = safe_execute(_fetch_market_status, fallback_value=None)
    
    if result is None:
        error_msg = "无法获取 Polygon 市场状态，请稍后重试"
        logger.warning(error_msg)
        return f"数据获取失败: {error_msg}"
    
    return result