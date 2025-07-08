#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据源配置管理工具
提供命令行界面来管理数据源配置
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tradingagents.dataflows.data_source_config import DataSourceConfig

# 获取配置管理专用日志器
logger = logging.getLogger('tradingagents.dataflows.config_manager')

def show_status(config_manager: DataSourceConfig):
    """显示当前配置状态"""
    logger.info("\n=== 数据源配置状态 ===")
    
    enabled_sources = config_manager.get_enabled_sources()
    disabled_sources = {}
    
    all_sources = config_manager.config.get('data_sources', {})
    for name, config in all_sources.items():
        if not config.get('enabled', False):
            disabled_sources[name] = config
    
    logger.info(f"\n已启用的数据源 ({len(enabled_sources)})：")
    for name, config in sorted(enabled_sources.items(), key=lambda x: x[1].get('priority', 999)):
        priority = config.get('priority', '未设置')
        timeout = config.get('timeout', '未设置')
        has_key = bool(config.get('api_key') or config.get('token'))
        key_status = "✓" if has_key else "✗"
        logger.info(f"  - {name}: 优先级={priority}, 超时={timeout}s, API密钥={key_status}")
    
    logger.info(f"\n已禁用的数据源 ({len(disabled_sources)})：")
    for name, config in disabled_sources.items():
        has_key = bool(config.get('api_key') or config.get('token'))
        key_status = "✓" if has_key else "✗"
        reason = "缺少API密钥" if not has_key else "手动禁用"
        logger.info(f"  - {name}: API密钥={key_status} ({reason})")
    
    # 验证配置
    issues = config_manager.validate_config()
    if issues['errors'] or issues['warnings']:
        logger.info("\n=== 配置问题 ===")
        for error in issues['errors']:
            logger.error(f"  错误: {error}")
        for warning in issues['warnings']:
            logger.warning(f"  警告: {warning}")
    else:
        logger.info("\n✓ 配置验证通过")

def enable_source(config_manager: DataSourceConfig, source_name: str):
    """启用数据源"""
    all_sources = config_manager.config.get('data_sources', {})
    if source_name not in all_sources:
        logger.error(f"错误: 未找到数据源 '{source_name}'")
        logger.info(f"可用的数据源: {', '.join(all_sources.keys())}")
        return
    
    source_config = all_sources[source_name]
    has_key = bool(source_config.get('api_key') or source_config.get('token'))
    
    if not has_key and source_name in ['alpha_vantage', 'tushare', 'polygon']:
        logger.warning(f"警告: {source_name} 需要API密钥才能正常工作")
        logger.info("请在 config.yaml 中配置相应的API密钥")
        response = input("是否仍要启用? (y/N): ")
        if response.lower() != 'y':
            return
    
    config_manager.enable_data_source(source_name, True)
    logger.info(f"✓ 已启用数据源: {source_name}")

def disable_source(config_manager: DataSourceConfig, source_name: str):
    """禁用数据源"""
    config_manager.enable_data_source(source_name, False)
    logger.info(f"✓ 已禁用数据源: {source_name}")

def set_priority(config_manager: DataSourceConfig, source_name: str, priority: int):
    """设置数据源优先级"""
    config_manager.set_priority(source_name, priority)
    logger.info(f"✓ 已设置 {source_name} 优先级为: {priority}")

def show_help():
    """显示帮助信息"""
    print("""
数据源配置管理工具

用法:
  python config_manager.py [命令] [参数]

命令:
  status                    - 显示当前配置状态
  enable <source_name>      - 启用数据源
  disable <source_name>     - 禁用数据源
  priority <source_name> <priority> - 设置优先级 (数字越小优先级越高)
  help                      - 显示此帮助信息

可用的数据源:
  - yahoo_finance    (Yahoo Finance, 免费)
  - sina_finance     (新浪财经, 免费)
  - alpha_vantage    (Alpha Vantage, 需要API密钥)
  - tushare          (TuShare Pro, 需要Token)
  - polygon          (Polygon.io, 需要API密钥)

示例:
  python config_manager.py status
  python config_manager.py enable alpha_vantage
  python config_manager.py priority sina_finance 1
  python config_manager.py disable yahoo_finance

注意:
  - API密钥需要在 config.yaml 文件中配置
  - 优先级数字越小，优先级越高
  - 建议至少保持一个免费数据源启用
""")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'help':
        show_help()
        return
    
    # 创建配置管理器
    try:
        config_manager = DataSourceConfig()
    except Exception as e:
        logger.error(f"错误: 无法加载配置文件: {e}")
        return
    
    if command == 'status':
        show_status(config_manager)
    
    elif command == 'enable':
        if len(sys.argv) < 3:
            logger.error("错误: 请指定要启用的数据源名称")
            return
        enable_source(config_manager, sys.argv[2])
    
    elif command == 'disable':
        if len(sys.argv) < 3:
            logger.error("错误: 请指定要禁用的数据源名称")
            return
        disable_source(config_manager, sys.argv[2])
    
    elif command == 'priority':
        if len(sys.argv) < 4:
            logger.error("错误: 请指定数据源名称和优先级")
            return
        try:
            priority = int(sys.argv[3])
            set_priority(config_manager, sys.argv[2], priority)
        except ValueError:
            logger.error("错误: 优先级必须是数字")
    
    else:
        logger.error(f"错误: 未知命令 '{command}'")
        show_help()

if __name__ == "__main__":
    main()