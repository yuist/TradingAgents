# gets data/stats

import yfinance as yf
from typing import Annotated, Callable, Any, Optional
from pandas import DataFrame
import pandas as pd
from functools import wraps
import time
from requests.exceptions import RequestException, HTTPError, Timeout

from .utils import save_output, SavePathType, decorate_all_methods
from ..utils.logging_manager import get_logger
from ..utils.retry_utils import with_retry, safe_execute, get_error_message

# 配置日志
logger = get_logger('dataflow', 'yahoo_finance')


def init_ticker(func: Callable) -> Callable:
    """Decorator to initialize yf.Ticker and pass it to the function."""

    @wraps(func)
    def wrapper(symbol: Annotated[str, "ticker symbol"], *args, **kwargs) -> Any:
        ticker = yf.Ticker(symbol)
        return func(ticker, *args, **kwargs)

    return wrapper


def handle_yfinance_errors(func: Callable) -> Callable:
    """装饰器：处理yfinance相关的错误和重试机制"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        def _execute():
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                if 'rate limit' in error_msg or 'too many requests' in error_msg:
                    logger.warning(f"Rate limit hit for {func.__name__}, waiting before retry...")
                    time.sleep(5)  # 等待5秒后重试
                    raise e  # 重新抛出异常以触发重试机制
                elif 'timeout' in error_msg or 'connection' in error_msg:
                    logger.warning(f"Network timeout for {func.__name__}, retrying...")
                    time.sleep(2)
                    raise e
                else:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise e
        
        return safe_execute(
            _execute,
            max_retries=3,
            delay=2,
            fallback_value=None
        )
    
    return wrapper


@decorate_all_methods(init_ticker)
class YFinanceUtils:

    @handle_yfinance_errors
    def get_stock_data(
        symbol: Annotated[str, "ticker symbol"],
        start_date: Annotated[
            str, "start date for retrieving stock price data, YYYY-mm-dd"
        ],
        end_date: Annotated[
            str, "end date for retrieving stock price data, YYYY-mm-dd"
        ],
        save_path: SavePathType = None,
    ) -> DataFrame:
        """retrieve stock price data for designated ticker symbol"""
        ticker = symbol
        # add one day to the end_date so that the data range is inclusive
        end_date = pd.to_datetime(end_date) + pd.DateOffset(days=1)
        end_date = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"Fetching stock data for {ticker.ticker} from {start_date} to {end_date}")
        
        # 使用超时机制获取数据
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        
        def fetch_data():
            return ticker.history(start=start_date, end=end_date)
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fetch_data)
                stock_data = future.result(timeout=30)  # 30秒超时
        except FutureTimeoutError:
            logger.error(f"Timeout fetching data for {ticker.ticker}")
            raise TimeoutError(f"Yahoo Finance请求超时（30秒）: {ticker.ticker}")
        
        if stock_data.empty:
            logger.warning(f"No data returned for {ticker.ticker}")
            return pd.DataFrame()  # 返回空DataFrame而不是None
        
        logger.info(f"Successfully fetched {len(stock_data)} records for {ticker.ticker}")
        # save_output(stock_data, f"Stock data for {ticker.ticker}", save_path)
        return stock_data

    @handle_yfinance_errors
    def get_stock_info(
        symbol: Annotated[str, "ticker symbol"],
    ) -> dict:
        """Fetches and returns latest stock information."""
        ticker = symbol
        logger.info(f"Fetching stock info for {ticker.ticker}")
        stock_info = ticker.info
        
        if not stock_info:
            logger.warning(f"No info returned for {ticker.ticker}")
            return {}  # 返回空字典而不是None
        
        logger.info(f"Successfully fetched info for {ticker.ticker}")
        return stock_info

    @handle_yfinance_errors
    def get_company_info(
        symbol: Annotated[str, "ticker symbol"],
        save_path: Optional[str] = None,
    ) -> DataFrame:
        """Fetches and returns company information as a DataFrame."""
        ticker = symbol
        logger.info(f"Fetching company info for {ticker.ticker}")
        info = ticker.info
        
        if not info:
            logger.warning(f"No company info returned for {ticker.ticker}")
            return pd.DataFrame()  # 返回空DataFrame
        
        company_info = {
            "Company Name": info.get("shortName", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Country": info.get("country", "N/A"),
            "Website": info.get("website", "N/A"),
        }
        company_info_df = DataFrame([company_info])
        
        if save_path:
            company_info_df.to_csv(save_path)
            logger.info(f"Company info for {ticker.ticker} saved to {save_path}")
        
        logger.info(f"Successfully fetched company info for {ticker.ticker}")
        return company_info_df

    @handle_yfinance_errors
    def get_stock_dividends(
        symbol: Annotated[str, "ticker symbol"],
        save_path: Optional[str] = None,
    ) -> DataFrame:
        """Fetches and returns the latest dividends data as a DataFrame."""
        ticker = symbol
        logger.info(f"Fetching dividends for {ticker.ticker}")
        dividends = ticker.dividends
        
        if dividends.empty:
            logger.warning(f"No dividends data returned for {ticker.ticker}")
            return pd.DataFrame()  # 返回空DataFrame
        
        if save_path:
            dividends.to_csv(save_path)
            logger.info(f"Dividends for {ticker.ticker} saved to {save_path}")
        
        logger.info(f"Successfully fetched dividends for {ticker.ticker}")
        return dividends

    @handle_yfinance_errors
    def get_income_stmt(symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """Fetches and returns the latest income statement of the company as a DataFrame."""
        ticker = symbol
        logger.info(f"Fetching income statement for {ticker.ticker}")
        income_stmt = ticker.financials
        
        if income_stmt.empty:
            logger.warning(f"No income statement data returned for {ticker.ticker}")
            return pd.DataFrame()  # 返回空DataFrame
        
        logger.info(f"Successfully fetched income statement for {ticker.ticker}")
        return income_stmt

    @handle_yfinance_errors
    def get_balance_sheet(symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """Fetches and returns the latest balance sheet of the company as a DataFrame."""
        ticker = symbol
        logger.info(f"Fetching balance sheet for {ticker.ticker}")
        balance_sheet = ticker.balance_sheet
        
        if balance_sheet.empty:
            logger.warning(f"No balance sheet data returned for {ticker.ticker}")
            return pd.DataFrame()  # 返回空DataFrame
        
        logger.info(f"Successfully fetched balance sheet for {ticker.ticker}")
        return balance_sheet

    @handle_yfinance_errors
    def get_cash_flow(symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """Fetches and returns the latest cash flow statement of the company as a DataFrame."""
        ticker = symbol
        logger.info(f"Fetching cash flow for {ticker.ticker}")
        cash_flow = ticker.cashflow
        
        if cash_flow.empty:
            logger.warning(f"No cash flow data returned for {ticker.ticker}")
            return pd.DataFrame()  # 返回空DataFrame
        
        logger.info(f"Successfully fetched cash flow for {ticker.ticker}")
        return cash_flow

    @handle_yfinance_errors
    def get_analyst_recommendations(symbol: Annotated[str, "ticker symbol"]) -> tuple:
        """Fetches the latest analyst recommendations and returns the most common recommendation and its count."""
        ticker = symbol
        logger.info(f"Fetching analyst recommendations for {ticker.ticker}")
        recommendations = ticker.recommendations
        
        if recommendations is None or recommendations.empty:
            logger.warning(f"No analyst recommendations returned for {ticker.ticker}")
            return None, 0  # No recommendations available

        try:
            # Assuming 'period' column exists and needs to be excluded
            row_0 = recommendations.iloc[0, 1:]  # Exclude 'period' column if necessary

            # Find the maximum voting result
            max_votes = row_0.max()
            majority_voting_result = row_0[row_0 == max_votes].index.tolist()
            
            logger.info(f"Successfully fetched analyst recommendations for {ticker.ticker}")
            return majority_voting_result[0], max_votes
        except Exception as e:
            logger.error(f"Error processing analyst recommendations for {ticker.ticker}: {e}")
            return None, 0
