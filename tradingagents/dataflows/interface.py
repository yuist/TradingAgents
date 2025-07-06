from typing import Annotated, Dict
from .reddit_utils import fetch_top_from_category
from .yfin_utils import *
from .stockstats_utils import *
from .googlenews_utils import *
from .finnhub_utils import get_data_in_range
from .alpha_vantage_utils import (
    get_stock_data_alpha_vantage,
    get_company_fundamentals_alpha_vantage,
    get_technical_indicators_alpha_vantage
)
from .polygon_utils import (
    get_stock_data_polygon,
    get_company_news_polygon,
    get_company_financials_polygon,
    get_market_status_polygon
)
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import pandas as pd
from tqdm import tqdm
import yfinance as yf
from openai import OpenAI
import time
# 移除对旧配置系统的依赖，现在使用主配置管理器
from ..utils.retry_utils import with_retry, safe_execute, get_error_message, with_yahoo_finance_retry
from ..config_manager import get_config_manager
from .qwen_config import QwenConfigManager
from ..utils.logging_manager import get_logger

# 配置日志
logger = get_logger('dataflow', 'interface')

def _get_data_dir():
    """获取数据目录路径"""
    from ..config_manager import get_config_manager
    config_manager = get_config_manager()
    return config_manager.config.get('data_dir', 'data')


def get_finnhub_news(
    ticker: Annotated[
        str,
        "Search query of a company's, e.g. 'AAPL, TSM, etc.",
    ],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
):
    """
    Retrieve news about a company within a time frame

    Args
        ticker (str): ticker for the company you are interested in
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns
        str: dataframe containing the news of the company in the time frame

    """

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    result = get_data_in_range(ticker, before, curr_date, "news_data", _get_data_dir())

    if len(result) == 0:
        return ""

    combined_result = ""
    for day, data in result.items():
        if len(data) == 0:
            continue
        for entry in data:
            current_news = (
                "### " + entry["headline"] + f" ({day})" + "\n" + entry["summary"]
            )
            combined_result += current_news + "\n\n"

    return f"## {ticker} News, from {before} to {curr_date}:\n" + str(combined_result)


def get_finnhub_company_insider_sentiment(
    ticker: Annotated[str, "ticker symbol for the company"],
    curr_date: Annotated[
        str,
        "current date of you are trading at, yyyy-mm-dd",
    ],
    look_back_days: Annotated[int, "number of days to look back"],
):
    """
    Retrieve insider sentiment about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading on, yyyy-mm-dd
    Returns:
        str: a report of the sentiment in the past 15 days starting at curr_date
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    data = get_data_in_range(ticker, before, curr_date, "insider_senti", _get_data_dir())

    if len(data) == 0:
        return ""

    result_str = ""
    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### {entry['year']}-{entry['month']}:\nChange: {entry['change']}\nMonthly Share Purchase Ratio: {entry['mspr']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} Insider Sentiment Data for {before} to {curr_date}:\n"
        + result_str
        + "The change field refers to the net buying/selling from all insiders' transactions. The mspr field refers to monthly share purchase ratio."
    )


def get_finnhub_company_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[
        str,
        "current date you are trading at, yyyy-mm-dd",
    ],
    look_back_days: Annotated[int, "how many days to look back"],
):
    """
    Retrieve insider transcaction information about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
        str: a report of the company's insider transaction/trading informtaion in the past 15 days
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    data = get_data_in_range(ticker, before, curr_date, "insider_trans", _get_data_dir())

    if len(data) == 0:
        return ""

    result_str = ""

    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### Filing Date: {entry['filingDate']}, {entry['name']}:\nChange:{entry['change']}\nShares: {entry['share']}\nTransaction Price: {entry['transactionPrice']}\nTransaction Code: {entry['transactionCode']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} insider transactions from {before} to {curr_date}:\n"
        + result_str
        + "The change field reflects the variation in share count—here a negative number indicates a reduction in holdings—while share specifies the total number of shares involved. The transactionPrice denotes the per-share price at which the trade was executed, and transactionDate marks when the transaction occurred. The name field identifies the insider making the trade, and transactionCode (e.g., S for sale) clarifies the nature of the transaction. FilingDate records when the transaction was officially reported, and the unique id links to the specific SEC filing, as indicated by the source. Additionally, the symbol ties the transaction to a particular company, isDerivative flags whether the trade involves derivative securities, and currency notes the currency context of the transaction."
    )


def get_simfin_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        _get_data_dir(),
        "fundamental_data",
        "simfin_data_all",
        "balance_sheet",
        "companies",
        "us",
        f"us-balance-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        logger.warning(
            "No balance sheet available before the given current date",
            extra={
                'ticker': ticker,
                'freq': freq,
                'curr_date': curr_date,
                'component': 'interface',
                'function': 'get_simfin_balance_sheet'
            }
        )
        return ""

    # Get the most recent balance sheet by selecting the row with the latest Publish Date
    latest_balance_sheet = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_balance_sheet = latest_balance_sheet.drop("SimFinId")

    return (
        f"## {freq} balance sheet for {ticker} released on {str(latest_balance_sheet['Publish Date'])[0:10]}: \n"
        + str(latest_balance_sheet)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of assets, liabilities, and equity. Assets are grouped as current (liquid items like cash and receivables) and noncurrent (long-term investments and property). Liabilities are split between short-term obligations and long-term debts, while equity reflects shareholder funds such as paid-in capital and retained earnings. Together, these components ensure that total assets equal the sum of liabilities and equity."
    )


def get_simfin_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        _get_data_dir(),
        "fundamental_data",
        "simfin_data_all",
        "cash_flow",
        "companies",
        "us",
        f"us-cashflow-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        logger.warning(
            "No cash flow statement available before the given current date",
            extra={
                'component': 'interface',
                'function': 'get_simfin_cashflow',
                'ticker': ticker,
                'freq': freq,
                'curr_date': curr_date
            }
        )
        return ""

    # Get the most recent cash flow statement by selecting the row with the latest Publish Date
    latest_cash_flow = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_cash_flow = latest_cash_flow.drop("SimFinId")

    return (
        f"## {freq} cash flow statement for {ticker} released on {str(latest_cash_flow['Publish Date'])[0:10]}: \n"
        + str(latest_cash_flow)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of cash movements. Operating activities show cash generated from core business operations, including net income adjustments for non-cash items and working capital changes. Investing activities cover asset acquisitions/disposals and investments. Financing activities include debt transactions, equity issuances/repurchases, and dividend payments. The net change in cash represents the overall increase or decrease in the company's cash position during the reporting period."
    )


def get_simfin_income_statements(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        _get_data_dir(),
        "fundamental_data",
        "simfin_data_all",
        "income_statements",
        "companies",
        "us",
        f"us-income-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        logger.warning(
            "No income statement available before the given current date",
            extra={
                'component': 'interface',
                'function': 'get_simfin_income_statements',
                'ticker': ticker,
                'freq': freq,
                'curr_date': curr_date
            }
        )
        return ""

    # Get the most recent income statement by selecting the row with the latest Publish Date
    latest_income = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_income = latest_income.drop("SimFinId")

    return (
        f"## {freq} income statement for {ticker} released on {str(latest_income['Publish Date'])[0:10]}: \n"
        + str(latest_income)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a comprehensive breakdown of the company's financial performance. Starting with Revenue, it shows Cost of Revenue and resulting Gross Profit. Operating Expenses are detailed, including SG&A, R&D, and Depreciation. The statement then shows Operating Income, followed by non-operating items and Interest Expense, leading to Pretax Income. After accounting for Income Tax and any Extraordinary items, it concludes with Net Income, representing the company's bottom-line profit or loss for the period."
    )


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    query = query.replace(" ", "+")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    news_results = getNewsData(query, before, curr_date)

    news_str = ""

    for news in news_results:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## {query} Google News, from {before} to {curr_date}:\n\n{news_str}"


def get_reddit_global_news(
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
    max_limit_per_day: Annotated[int, "Maximum number of news per day"],
) -> str:
    """
    Retrieve the latest top reddit news
    Args:
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the latest news articles posts on reddit and meta information in these columns: "created_utc", "id", "title", "selftext", "score", "num_comments", "url"
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    posts = []
    # iterate from start_date to end_date
    curr_date = datetime.strptime(before, "%Y-%m-%d")

    total_iterations = (start_date - curr_date).days + 1
    pbar = tqdm(desc=f"Getting Global News on {start_date}", total=total_iterations)

    while curr_date <= start_date:
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        fetch_result = fetch_top_from_category(
            "global_news",
            curr_date_str,
            max_limit_per_day,
            data_path=os.path.join(_get_data_dir(), "reddit_data"),
        )
        posts.extend(fetch_result)
        curr_date += relativedelta(days=1)
        pbar.update(1)

    pbar.close()

    if len(posts) == 0:
        return ""

    news_str = ""
    for post in posts:
        if post["content"] == "":
            news_str += f"### {post['title']}\n\n"
        else:
            news_str += f"### {post['title']}\n\n{post['content']}\n\n"

    return f"## Global News Reddit, from {before} to {curr_date}:\n{news_str}"


def get_reddit_company_news(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
    max_limit_per_day: Annotated[int, "Maximum number of news per day"],
) -> str:
    """
    Retrieve the latest top reddit news
    Args:
        ticker: ticker symbol of the company
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the latest news articles posts on reddit and meta information in these columns: "created_utc", "id", "title", "selftext", "score", "num_comments", "url"
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    posts = []
    # iterate from start_date to end_date
    curr_date = datetime.strptime(before, "%Y-%m-%d")

    total_iterations = (start_date - curr_date).days + 1
    pbar = tqdm(
        desc=f"Getting Company News for {ticker} on {start_date}",
        total=total_iterations,
    )

    while curr_date <= start_date:
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        fetch_result = fetch_top_from_category(
            "company_news",
            curr_date_str,
            max_limit_per_day,
            ticker,
            data_path=os.path.join(_get_data_dir(), "reddit_data"),
        )
        posts.extend(fetch_result)
        curr_date += relativedelta(days=1)

        pbar.update(1)

    pbar.close()

    if len(posts) == 0:
        return ""

    news_str = ""
    for post in posts:
        if post["content"] == "":
            news_str += f"### {post['title']}\n\n"
        else:
            news_str += f"### {post['title']}\n\n{post['content']}\n\n"

    return f"##{ticker} News Reddit, from {before} to {curr_date}:\n\n{news_str}"


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date - relativedelta(days=look_back_days)

    if not online:
        # read from YFin data
        data = pd.read_csv(
            os.path.join(
                _get_data_dir(),
                f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
            )
        )
        data["Date"] = pd.to_datetime(data["Date"], utc=True)
        dates_in_df = data["Date"].astype(str).str[:10]

        ind_string = ""
        while curr_date >= before:
            # only do the trading dates
            if curr_date.strftime("%Y-%m-%d") in dates_in_df.values:
                indicator_value = get_stockstats_indicator(
                    symbol, indicator, curr_date.strftime("%Y-%m-%d"), online
                )

                ind_string += f"{curr_date.strftime('%Y-%m-%d')}: {indicator_value}\n"

            curr_date = curr_date - relativedelta(days=1)
    else:
        # online gathering
        ind_string = ""
        while curr_date >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date.strftime("%Y-%m-%d"), online
            )

            ind_string += f"{curr_date.strftime('%Y-%m-%d')}: {indicator_value}\n"

            curr_date = curr_date - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:

    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
            os.path.join(_get_data_dir(), "market_data", "price_data"),
            online=online,
        )
    except Exception as e:
        logger.error(
            f"Error getting stockstats indicator data for {symbol} {indicator} on {curr_date}: {type(e).__name__}: {str(e)}",
            extra={
                'component': 'interface',
                'function': 'get_stockstats_indicator',
                'symbol': symbol,
                'indicator': indicator,
                'curr_date': curr_date,
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
        )
        return ""

    return str(indicator_value)


def get_YFin_data_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    # calculate past days
    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    start_date = before.strftime("%Y-%m-%d")

    # read in data
    data = pd.read_csv(
        os.path.join(
            _get_data_dir(),
            f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
        )
    )

    # Extract just the date part for comparison
    data["DateOnly"] = data["Date"].str[:10]

    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["DateOnly"] >= start_date) & (data["DateOnly"] <= curr_date)
    ]

    # Drop the temporary column we created
    filtered_data = filtered_data.drop("DateOnly", axis=1)

    # Set pandas display options to show the full DataFrame
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        df_string = filtered_data.to_string()

    return (
        f"## Raw Market Data for {symbol} from {start_date} to {curr_date}:\n\n"
        + df_string
    )


@with_yahoo_finance_retry(
    max_retries=3,
    initial_delay=60.0,
    max_delay=600.0,
    backoff_factor=2.0,
    rate_limit_interval=30.0
)
def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    quick_fail: bool = False,
):
    """获取股票数据，使用多数据源系统提供备用支持
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        quick_fail: 快速失败模式，减少等待时间
    """
    
    # 验证日期格式
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")
    
    mode_desc = "快速失败模式" if quick_fail else "标准模式"
    logger.info(f"正在获取 {symbol} 的股票数据，时间范围: {start_date} 到 {end_date}，模式: {mode_desc}")
    
    # 快速失败模式：跳过多数据源系统，直接使用简化的Yahoo Finance
    if quick_fail:
        logger.info("快速失败模式：直接使用Yahoo Finance，跳过多数据源系统")
    else:
        try:
            # 尝试使用增强的多数据源接口
            from .enhanced_interface import get_YFin_data_online_enhanced
            result = get_YFin_data_online_enhanced(symbol, start_date, end_date)
            
            # 如果增强接口返回有效数据，直接返回
            if result and not result.startswith("数据获取失败") and not result.startswith("未找到"):
                logger.info(f"通过多数据源系统成功获取 {symbol} 的股票数据")
                return result
            else:
                logger.warning(f"多数据源系统未能获取数据，回退到原始Yahoo Finance接口")
                
        except Exception as e:
            logger.warning(f"多数据源系统调用失败: {e}，回退到原始Yahoo Finance接口")
    
    # 回退到原始的Yahoo Finance实现
    try:
        import yfinance as yf
        
        # Create ticker object
        ticker = yf.Ticker(symbol.upper())
        
        # 在快速失败模式下设置更短的超时时间
        if quick_fail:
            logger.info("快速失败模式：使用5秒超时")
            # 使用线程超时机制（Windows兼容）
            import threading
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
            
            def fetch_data():
                return ticker.history(start=start_date, end=end_date)
            
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(fetch_data)
                    data = future.result(timeout=5)  # 5秒超时
            except FutureTimeoutError:
                raise TimeoutError("Yahoo Finance请求超时（快速失败模式，5秒）")
        else:
            # Fetch historical data for the specified date range with timeout
            import threading
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
            
            def fetch_data():
                return ticker.history(start=start_date, end=end_date)
            
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(fetch_data)
                    data = future.result(timeout=30)  # 30秒超时
            except FutureTimeoutError:
                raise TimeoutError("Yahoo Finance请求超时（标准模式，30秒）")
        
        # Check if data is empty
        if data.empty:
            warning_msg = f"未找到股票代码 '{symbol}' 在 {start_date} 到 {end_date} 期间的数据"
            logger.warning(warning_msg)
            return warning_msg
        
        # Remove timezone info from index for cleaner output
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # Round numerical values to 2 decimal places for cleaner display
        numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
        for col in numeric_columns:
            if col in data.columns:
                data[col] = data[col].round(2)
        
        # Convert DataFrame to CSV string
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
        header += f"# Total records: {len(data)}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"# Data source: Yahoo Finance (fallback)\n\n"
        
        logger.info(f"通过Yahoo Finance回退接口成功获取 {symbol} 的股票数据，共 {len(data)} 条记录")
        return header + csv_string
        
    except Exception as e:
        error_msg = f"所有数据源均失败，无法获取 {symbol} 的股票数据: {str(e)}"
        logger.error(error_msg)
        return f"数据获取失败: {error_msg}"


def get_YFin_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    # read in data
    data = pd.read_csv(
        os.path.join(
            _get_data_dir(),
            f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
        )
    )

    if end_date > "2025-03-25":
        raise Exception(
            f"Get_YFin_Data: {end_date} is outside of the data range of 2015-01-01 to 2025-03-25"
        )

    # Extract just the date part for comparison
    data["DateOnly"] = data["Date"].str[:10]

    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["DateOnly"] >= start_date) & (data["DateOnly"] <= end_date)
    ]

    # Drop the temporary column we created
    filtered_data = filtered_data.drop("DateOnly", axis=1)

    # remove the index from the dataframe
    filtered_data = filtered_data.reset_index(drop=True)

    return filtered_data


@with_retry(max_retries=3, delay=2)
def get_stock_news_openai(ticker: str, curr_date: str) -> str:
    """使用LLM Web Search获取股票新闻数据
    
    支持的提供商:
    - OpenAI: 使用web_search_options参数
    - Qwen: 使用enable_search参数
    
    Args:
        ticker: 股票代码
        curr_date: 当前日期
        
    Returns:
        str: 新闻内容或错误信息
    """
    logger.info(f"正在使用LLM Web Search获取股票 {ticker} 的新闻数据，日期: {curr_date}")
    
    def _fetch_news():
        from ..config_manager import get_config_manager
        config_manager = get_config_manager()
        llm_config = config_manager.get_llm_config()
        
        # 验证配置
        if not llm_config.api_key:
            error_msg = f"API密钥未配置，提供商: {config_manager.config['models']['llm_provider']}"
            logger.error(error_msg)
            raise ValueError(f"配置错误: {error_msg}")
        
        client = OpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            timeout=120,
            max_retries=3
        )

        provider = llm_config.provider
        model = llm_config.quick_think_model
        
        # 根据提供商配置不同的搜索参数
        if provider == "qwen":
            # 使用QwenConfigManager获取优化的配置
            extra_body_config = QwenConfigManager.get_news_search_config(
                model=model,
                task_type="stock_news"
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Search for recent news and social media discussions about {ticker} stock from 7 days before {curr_date} to {curr_date}. Focus on news that could impact stock price and trading decisions. Please provide sources and dates."
                    }
                ],
                temperature=0.1,
                max_tokens=4096,
                extra_body=extra_body_config
            )
        else:
            # OpenAI等其他提供商使用web_search_options
            if provider == "openai":
                model = "gpt-4o-search-preview"  # OpenAI专用搜索模型
            
            response = client.chat.completions.create(
                model=model,
                web_search_options={},  # 启用web search
                messages=[
                    {
                        "role": "user",
                        "content": f"Search for recent news and social media discussions about {ticker} stock from 7 days before {curr_date} to {curr_date}. Focus on news that could impact stock price and trading decisions. Please provide sources and dates."
                    }
                ],
                temperature=0.1,
                max_tokens=4096
            )

        # 验证响应结构
        if not response.choices or len(response.choices) == 0:
            error_msg = "API返回了空响应"
            logger.warning(error_msg)
            raise ValueError(f"数据获取失败: {error_msg}")
            
        content = response.choices[0].message.content
        if not content:
            error_msg = "API返回的内容为空"
            logger.warning(error_msg)
            raise ValueError(f"数据获取失败: {error_msg}")
            
        logger.info(f"成功获取股票 {ticker} 的新闻数据，内容长度: {len(content)}")
        return content
    
    # 使用安全执行函数
    result = safe_execute(_fetch_news, fallback_value=None)
    
    if result is None:
        error_msg = f"无法获取股票 {ticker} 的新闻数据，请稍后重试"
        logger.warning(error_msg)
        return f"数据获取失败: {error_msg}"
    
    return result


@with_retry(max_retries=3, delay=2)
def get_global_news_openai(curr_date: str) -> str:
    """使用OpenAI Web Search获取全球新闻数据
    
    Args:
        curr_date: 当前日期
        
    Returns:
        str: 新闻内容或错误信息
    """
    logger.info(f"正在使用OpenAI Web Search获取全球新闻数据，日期: {curr_date}")
    
    def _fetch_global_news():
        from ..config_manager import get_config_manager
        config_manager = get_config_manager()
        llm_config = config_manager.get_llm_config()
        
        # 验证配置
        if not llm_config.api_key:
            error_msg = f"API密钥未配置，提供商: {config_manager.config['models']['llm_provider']}"
            logger.error(error_msg)
            raise ValueError(f"配置错误: {error_msg}")
        
        client = OpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            timeout=120,
            max_retries=3
        )

        provider = llm_config.provider
        model = llm_config.quick_think_model
        
        # 根据提供商配置不同的搜索参数
        if provider == "qwen":
            # 使用QwenConfigManager获取优化的配置
            extra_body_config = QwenConfigManager.get_news_search_config(
                model=model,
                task_type="global_news"
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Search for global macroeconomic news and market-moving events from 7 days before {curr_date} to {curr_date}. Focus on central bank decisions, economic indicators, geopolitical events, and other news that could impact financial markets. Please provide sources and dates."
                    }
                ],
                temperature=0.1,
                max_tokens=4096,
                extra_body=extra_body_config
            )
        else:
            # OpenAI等其他提供商使用web_search_options
            if provider == "openai":
                model = "gpt-4o-search-preview"  # OpenAI专用搜索模型
            
            response = client.chat.completions.create(
                model=model,
                web_search_options={},  # 启用web search
                messages=[
                    {
                        "role": "user",
                        "content": f"Search for global macroeconomic news and market-moving events from 7 days before {curr_date} to {curr_date}. Focus on central bank decisions, economic indicators, geopolitical events, and other news that could impact financial markets. Please provide sources and dates."
                    }
                ],
                temperature=0.1,
                max_tokens=4096
            )

        # 验证响应结构
        if not response.choices or len(response.choices) == 0:
            error_msg = "API返回了空响应"
            logger.warning(error_msg)
            raise ValueError(f"数据获取失败: {error_msg}")
            
        content = response.choices[0].message.content
        if not content:
            error_msg = "API返回的内容为空"
            logger.warning(error_msg)
            raise ValueError(f"数据获取失败: {error_msg}")
            
        logger.info(f"成功获取全球新闻数据，内容长度: {len(content)}")
        return content
    
    # 使用安全执行函数
    result = safe_execute(_fetch_global_news, fallback_value=None)
    
    if result is None:
        error_msg = "无法获取全球新闻数据，请稍后重试"
        logger.warning(error_msg)
        return f"数据获取失败: {error_msg}"
    
    return result


@with_retry(max_retries=3, delay=2)
def get_fundamentals_openai(ticker: str, curr_date: str) -> str:
    """获取基本面数据
    
    Args:
        ticker: 股票代码
        curr_date: 当前日期
        
    Returns:
        str: 基本面数据或错误信息
    """
    logger.info(f"正在获取股票 {ticker} 的基本面数据，日期: {curr_date}")
    
    def _fetch_fundamentals():
        from ..config_manager import get_config_manager
        config_manager = get_config_manager()
        llm_config = config_manager.get_llm_config()
        
        # 验证配置
        if not llm_config.api_key:
            error_msg = f"API密钥未配置，提供商: {config_manager.config['models']['llm_provider']}"
            logger.error(error_msg)
            raise ValueError(f"配置错误: {error_msg}")
        
        client = OpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            timeout=120,
            max_retries=3
        )

        response = client.chat.completions.create(
            model=config_manager.config['models']['quick_think_llm'],
            messages=[
                {
                    "role": "user",
                    "content": f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date} to the month of {curr_date}. Make sure you only get the data posted during that period. List as a table, with PE/PS/Cash flow/ etc"
                }
            ],
            temperature=1,
            max_tokens=4096,
            top_p=1
        )

        # 验证响应结构
        if not response.choices or len(response.choices) == 0:
            error_msg = "API返回了空响应"
            logger.warning(error_msg)
            raise ValueError(f"数据获取失败: {error_msg}")
            
        content = response.choices[0].message.content
        if not content:
            error_msg = "API返回的内容为空"
            logger.warning(error_msg)
            raise ValueError(f"数据获取失败: {error_msg}")
            
        logger.info(f"成功获取股票 {ticker} 的基本面数据，内容长度: {len(content)}")
        return content
    
    # 使用安全执行函数
    result = safe_execute(_fetch_fundamentals, fallback_value=None)
    
    if result is None:
        error_msg = f"无法获取股票 {ticker} 的基本面数据，请稍后重试"
        logger.warning(error_msg)
        return f"数据获取失败: {error_msg}"
    
    return result


# Alpha Vantage 数据接口函数
def get_alpha_vantage_stock_data(
    ticker: Annotated[str, "股票代码，如 'AAPL', 'TSLA' 等"],
    curr_date: Annotated[str, "当前日期，格式为 yyyy-mm-dd"],
    look_back_days: Annotated[int, "回溯天数，默认30天"] = 30
) -> str:
    """
    获取 Alpha Vantage 股票数据
    
    Args:
        ticker: 股票代码
        curr_date: 当前日期
        look_back_days: 回溯天数
        
    Returns:
        str: 格式化的股票数据
    """
    from datetime import datetime, timedelta
    
    # 计算开始日期
    end_date = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=look_back_days)
    start_date_str = start_date.strftime("%Y-%m-%d")
    
    return get_stock_data_alpha_vantage(ticker, start_date_str, curr_date)


def get_alpha_vantage_fundamentals(
    ticker: Annotated[str, "股票代码，如 'AAPL', 'TSLA' 等"],
    curr_date: Annotated[str, "当前日期，格式为 yyyy-mm-dd"]
) -> str:
    """
    获取 Alpha Vantage 公司基本面数据
    
    Args:
        ticker: 股票代码
        curr_date: 当前日期
        
    Returns:
        str: 格式化的基本面数据
    """
    return get_company_fundamentals_alpha_vantage(ticker)


def get_alpha_vantage_technical_indicators(
    ticker: Annotated[str, "股票代码，如 'AAPL', 'TSLA' 等"],
    curr_date: Annotated[str, "当前日期，格式为 yyyy-mm-dd"],
    indicator: Annotated[str, "技术指标类型，如 'SMA', 'EMA', 'RSI', 'MACD' 等"] = "SMA",
    time_period: Annotated[int, "时间周期，默认20"] = 20
) -> str:
    """
    获取 Alpha Vantage 技术指标数据
    
    Args:
        ticker: 股票代码
        curr_date: 当前日期
        indicator: 技术指标类型
        time_period: 时间周期
        
    Returns:
        str: 格式化的技术指标数据
    """
    return get_technical_indicators_alpha_vantage(ticker, indicator, time_period)


# Polygon 数据接口函数
def get_polygon_stock_data(
    ticker: Annotated[str, "股票代码，如 'AAPL', 'TSLA' 等"],
    curr_date: Annotated[str, "当前日期，格式为 yyyy-mm-dd"],
    look_back_days: Annotated[int, "回溯天数，默认30天"] = 30
) -> str:
    """
    获取 Polygon 股票数据
    
    Args:
        ticker: 股票代码
        curr_date: 当前日期
        look_back_days: 回溯天数
        
    Returns:
        str: 格式化的股票数据
    """
    from datetime import datetime, timedelta
    
    # 计算开始日期
    end_date = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=look_back_days)
    start_date_str = start_date.strftime("%Y-%m-%d")
    
    return get_stock_data_polygon(ticker, start_date_str, curr_date)


def get_polygon_company_news(
    ticker: Annotated[str, "股票代码，如 'AAPL', 'TSLA' 等"],
    curr_date: Annotated[str, "当前日期，格式为 yyyy-mm-dd"],
    look_back_days: Annotated[int, "回溯天数，默认7天"] = 7,
    limit: Annotated[int, "新闻数量限制，默认10条"] = 10
) -> str:
    """
    获取 Polygon 公司新闻
    
    Args:
        ticker: 股票代码
        curr_date: 当前日期
        look_back_days: 回溯天数
        limit: 新闻数量限制
        
    Returns:
        str: 格式化的新闻数据
    """
    from datetime import datetime, timedelta
    
    # 计算开始日期
    end_date = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=look_back_days)
    start_date_str = start_date.strftime("%Y-%m-%d")
    
    return get_company_news_polygon(ticker, start_date_str, curr_date, limit)


def get_polygon_company_financials(
    ticker: Annotated[str, "股票代码，如 'AAPL', 'TSLA' 等"],
    curr_date: Annotated[str, "当前日期，格式为 yyyy-mm-dd"],
    timeframe: Annotated[str, "时间框架，'annual' 或 'quarterly'"] = "annual"
) -> str:
    """
    获取 Polygon 公司财务数据
    
    Args:
        ticker: 股票代码
        curr_date: 当前日期
        timeframe: 时间框架
        
    Returns:
        str: 格式化的财务数据
    """
    return get_company_financials_polygon(ticker, timeframe)


def get_polygon_market_status(
    curr_date: Annotated[str, "当前日期，格式为 yyyy-mm-dd"]
) -> str:
    """
    获取 Polygon 市场状态
    
    Args:
        curr_date: 当前日期
        
    Returns:
        str: 格式化的市场状态
    """
    return get_market_status_polygon()
