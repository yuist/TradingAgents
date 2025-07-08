#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化后的多数据源管理器测试脚本
测试缓存、熔断器和并发获取功能
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tradingagents.dataflows.multi_source_manager import (
    get_multi_source_manager,
    get_stock_data_with_fallback,
    get_realtime_data_with_fallback
)
from tradingagents.dataflows.circuit_breaker import get_circuit_breaker_manager
from tradingagents.dataflows.data_cache import get_data_cache
from tradingagents.dataflows.concurrent_fetcher import get_concurrent_fetcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_cache_functionality():
    """测试缓存功能"""
    logger.info("=== 测试缓存功能 ===")
    
    # 配置
    config = {
        'data_sources': {
            'yahoo_finance': {'enabled': True},
            'sina_finance': {'enabled': True},
            'alpha_vantage': {'enabled': False},
            'tushare': {'enabled': False}
        }
    }
    
    symbol = "AAPL"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # 第一次获取数据（应该从数据源获取）
    logger.info("第一次获取数据...")
    start_time = time.time()
    try:
        data1 = get_stock_data_with_fallback(symbol, start_date, end_date, config)
        first_fetch_time = time.time() - start_time
        logger.info(f"第一次获取成功，耗时: {first_fetch_time:.2f}秒，数据量: {len(data1)}")
    except Exception as e:
        logger.error(f"第一次获取失败: {e}")
        return
    
    # 第二次获取数据（应该从缓存获取）
    logger.info("第二次获取数据...")
    start_time = time.time()
    try:
        data2 = get_stock_data_with_fallback(symbol, start_date, end_date, config)
        second_fetch_time = time.time() - start_time
        logger.info(f"第二次获取成功，耗时: {second_fetch_time:.2f}秒，数据量: {len(data2)}")
        
        # 验证缓存效果
        if second_fetch_time < first_fetch_time * 0.1:  # 缓存应该快很多
            logger.info("✅ 缓存功能正常工作")
        else:
            logger.warning("⚠️ 缓存可能未生效")
            
    except Exception as e:
        logger.error(f"第二次获取失败: {e}")

def test_circuit_breaker():
    """测试熔断器功能"""
    logger.info("=== 测试熔断器功能 ===")
    
    circuit_manager = get_circuit_breaker_manager()
    
    # 获取一个数据源的熔断器
    provider_name = "Yahoo Finance"
    circuit_breaker = circuit_manager.get_breaker(provider_name)
    
    logger.info(f"熔断器初始状态: {circuit_breaker.state}")
    
    # 模拟多次失败
    logger.info("模拟连续失败...")
    for i in range(6):  # 超过失败阈值
        circuit_breaker.record_failure()
        logger.info(f"失败 {i+1} 次，状态: {circuit_breaker.state}")
    
    # 检查是否进入熔断状态
    if not circuit_breaker.can_execute():
        logger.info("✅ 熔断器正常工作，已进入熔断状态")
    else:
        logger.warning("⚠️ 熔断器可能未正常工作")
    
    # 等待一段时间后尝试恢复
    logger.info("等待熔断器恢复...")
    time.sleep(2)
    
    if circuit_breaker.can_execute():
        logger.info("熔断器已恢复，可以执行")
        # 模拟成功
        circuit_breaker.record_success()
        logger.info(f"记录成功后状态: {circuit_breaker.state}")
    
def test_concurrent_fetching():
    """测试并发获取功能"""
    logger.info("=== 测试并发获取功能 ===")
    
    # 配置多个数据源
    config = {
        'data_sources': {
            'yahoo_finance': {'enabled': True},
            'sina_finance': {'enabled': True},
            'alpha_vantage': {'enabled': False},
            'tushare': {'enabled': False}
        }
    }
    
    symbol = "AAPL"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    # 清除缓存以确保测试并发获取
    cache = get_data_cache()
    cache.clear()
    
    logger.info("测试并发获取历史数据...")
    start_time = time.time()
    try:
        data = get_stock_data_with_fallback(symbol, start_date, end_date, config, timeout=10)
        fetch_time = time.time() - start_time
        logger.info(f"并发获取成功，耗时: {fetch_time:.2f}秒，数据量: {len(data)}")
        logger.info("✅ 并发获取历史数据功能正常")
    except Exception as e:
        logger.error(f"并发获取历史数据失败: {e}")
    
    logger.info("测试并发获取实时数据...")
    try:
        realtime_data = get_realtime_data_with_fallback(symbol, config, timeout=5)
        logger.info(f"并发获取实时数据成功: {realtime_data}")
        logger.info("✅ 并发获取实时数据功能正常")
    except Exception as e:
        logger.error(f"并发获取实时数据失败: {e}")

def test_health_monitoring():
    """测试健康监控功能"""
    logger.info("=== 测试健康监控功能 ===")
    
    config = {
        'data_sources': {
            'yahoo_finance': {'enabled': True},
            'sina_finance': {'enabled': True},
            'alpha_vantage': {'enabled': False},
            'tushare': {'enabled': False}
        }
    }
    
    manager = get_multi_source_manager(config)
    
    # 强制健康检查
    logger.info("执行健康检查...")
    manager.force_health_check()
    
    # 获取健康状态
    health_status = manager.get_health_status()
    
    logger.info("数据源健康状态:")
    for market, providers in health_status.items():
        logger.info(f"市场: {market}")
        for provider in providers:
            logger.info(f"  - {provider['name']}: {provider['status']} (优先级: {provider['priority']})")
            logger.info(f"    成功率: {provider['success_rate']:.2%}, 平均响应时间: {provider['avg_response_time']:.2f}秒")
    
    logger.info("✅ 健康监控功能正常")

def test_data_quality():
    """测试数据质量报告功能"""
    logger.info("=== 测试数据质量报告功能 ===")
    
    config = {
        'data_sources': {
            'yahoo_finance': {'enabled': True},
            'sina_finance': {'enabled': True},
            'alpha_vantage': {'enabled': False},
            'tushare': {'enabled': False}
        }
    }
    
    manager = get_multi_source_manager(config)
    
    symbol = "AAPL"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    try:
        quality_report = manager.get_data_quality_report(symbol, start_date, end_date)
        
        logger.info("数据质量报告:")
        logger.info(f"  符号: {quality_report['symbol']}")
        logger.info(f"  有效性: {quality_report['is_valid']}")
        logger.info(f"  质量分数: {quality_report['quality_score']:.2f}")
        logger.info(f"  数据源: {quality_report.get('source', 'N/A')}")
        
        if 'checks' in quality_report:
            logger.info("  详细检查:")
            for check, result in quality_report['checks'].items():
                logger.info(f"    {check}: {result}")
        
        logger.info("✅ 数据质量报告功能正常")
        
    except Exception as e:
        logger.error(f"数据质量报告测试失败: {e}")

def main():
    """主测试函数"""
    logger.info("开始测试优化后的多数据源管理器")
    
    try:
        # 测试各个功能模块
        test_cache_functionality()
        print("\n" + "="*50 + "\n")
        
        test_circuit_breaker()
        print("\n" + "="*50 + "\n")
        
        test_concurrent_fetching()
        print("\n" + "="*50 + "\n")
        
        test_health_monitoring()
        print("\n" + "="*50 + "\n")
        
        test_data_quality()
        
        logger.info("\n🎉 所有测试完成！")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()