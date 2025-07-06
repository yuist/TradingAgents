#!/usr/bin/env python3
"""
多数据提供商使用示例

本示例展示如何使用 TradingAgents 框架中集成的多个金融数据提供商：
- Finnhub: 新闻、内部人士情绪分析
- Alpha Vantage: 技术指标、基本面数据
- Polygon: 实时市场数据、财务报表
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows import (
    # Finnhub 数据
    get_finnhub_news,
    get_finnhub_company_insider_sentiment,
    
    # Alpha Vantage 数据
    get_alpha_vantage_stock_data,
    get_alpha_vantage_fundamentals,
    get_alpha_vantage_technical_indicators,
    
    # Polygon 数据
    get_polygon_stock_data,
    get_polygon_company_news,
    get_polygon_company_financials,
    get_polygon_market_status
)

def demonstrate_multi_provider_usage():
    """
    演示多数据提供商的综合使用
    """
    symbol = "AAPL"
    print(f"\n=== {symbol} 多数据源分析 ===")
    
    # 1. 使用 Finnhub 获取新闻和情绪数据
    print("\n1. Finnhub - 新闻和情绪分析")
    try:
        news = get_finnhub_news(symbol, days_back=7)
        if news:
            print(f"✓ 获取到 {len(news)} 条新闻")
        
        sentiment = get_finnhub_company_insider_sentiment(symbol)
        if sentiment:
            print("✓ 获取到内部人士情绪数据")
    except Exception as e:
        print(f"⚠ Finnhub 数据获取失败: {e}")
    
    # 2. 使用 Alpha Vantage 获取技术指标
    print("\n2. Alpha Vantage - 技术分析")
    try:
        # 获取股票价格数据
        stock_data = get_alpha_vantage_stock_data(symbol, function="TIME_SERIES_DAILY")
        if stock_data:
            print("✓ 获取到股票价格数据")
        
        # 获取技术指标 - RSI
        rsi_data = get_alpha_vantage_technical_indicators(
            symbol, "RSI", interval="daily", time_period=14
        )
        if rsi_data:
            print("✓ 获取到 RSI 技术指标")
        
        # 获取基本面数据
        fundamentals = get_alpha_vantage_fundamentals(symbol)
        if fundamentals:
            print("✓ 获取到基本面数据")
    except Exception as e:
        print(f"⚠ Alpha Vantage 数据获取失败: {e}")
    
    # 3. 使用 Polygon 获取实时数据和财务报表
    print("\n3. Polygon - 实时数据和财务")
    try:
        # 获取市场状态
        from datetime import datetime
        market_status = get_polygon_market_status(datetime.now().strftime("%Y-%m-%d"))
        if market_status:
            print("✓ 获取到市场状态")
        
        # 获取股票数据
        polygon_stock = get_polygon_stock_data(symbol, timespan="day", limit=10)
        if polygon_stock:
            print("✓ 获取到 Polygon 股票数据")
        
        # 获取公司新闻
        company_news = get_polygon_company_news(symbol, limit=5)
        if company_news:
            print(f"✓ 获取到 {len(company_news)} 条公司新闻")
        
        # 获取财务数据
        financials = get_polygon_company_financials(symbol, limit=4)
        if financials:
            print("✓ 获取到财务报表数据")
    except Exception as e:
        print(f"⚠ Polygon 数据获取失败: {e}")

def show_data_provider_strengths():
    """
    展示各数据提供商的优势和使用场景
    """
    print("\n=== 数据提供商优势对比 ===")
    
    providers = {
        "Finnhub": {
            "优势": ["丰富的新闻数据", "内部人士交易分析", "情绪指标", "免费额度充足"],
            "适用场景": ["新闻情绪分析", "内部人士行为追踪", "市场情绪监控"]
        },
        "Alpha Vantage": {
            "优势": ["全面的技术指标", "基本面数据", "经济指标", "历史数据完整"],
            "适用场景": ["技术分析", "基本面分析", "量化策略开发"]
        },
        "Polygon": {
            "优势": ["实时数据", "高频数据", "多资产类别", "详细财务报表"],
            "适用场景": ["实时交易", "高频策略", "财务分析", "多资产组合"]
        }
    }
    
    for provider, info in providers.items():
        print(f"\n{provider}:")
        print(f"  优势: {', '.join(info['优势'])}")
        print(f"  适用场景: {', '.join(info['适用场景'])}")

def main():
    """
    主函数
    """
    print("TradingAgents 多数据提供商使用示例")
    print("=" * 50)
    
    # 展示数据提供商优势
    show_data_provider_strengths()
    
    # 演示多提供商综合使用
    demonstrate_multi_provider_usage()
    
    print("\n=== 使用建议 ===")
    print("1. 根据分析需求选择合适的数据提供商")
    print("2. 结合多个数据源可以获得更全面的市场视角")
    print("3. 注意各提供商的 API 限制和成本")
    print("4. 在生产环境中建议实现数据缓存和错误处理")

if __name__ == "__main__":
    main()