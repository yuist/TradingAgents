#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多数据源配置管理
提供数据源的配置和管理功能
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

# 获取数据源配置专用日志器
logger = logging.getLogger('tradingagents.dataflows.config')

class DataSourceConfig:
    """数据源配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None, main_config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_path()
        self.main_config_file = main_config_file or self._get_main_config_path()
        self.config = self._load_config()
        self._load_api_keys_from_main_config()
    
    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        return os.path.join(os.path.dirname(__file__), 'data_sources.json')
    
    def _get_main_config_path(self) -> str:
        """获取主配置文件路径"""
        # 从当前文件向上查找项目根目录的config.yaml
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(current_dir))
        return os.path.join(project_root, 'config.yaml')
    
    def _load_api_keys_from_main_config(self):
        """从主配置文件加载API密钥"""
        if not os.path.exists(self.main_config_file):
            return
        
        try:
            with open(self.main_config_file, 'r', encoding='utf-8') as f:
                main_config = yaml.safe_load(f)
            
            api_keys = main_config.get('api_keys', {})
            
            # 确保config中有api_keys结构
            if 'api_keys' not in self.config:
                self.config['api_keys'] = {}
            
            # 将API密钥复制到顶层api_keys结构（供MultiSourceManager使用）
            self.config['api_keys'].update(api_keys)
            
            # 更新Alpha Vantage API密钥
            if 'alpha_vantage_api_key' in api_keys and api_keys['alpha_vantage_api_key']:
                if 'alpha_vantage' in self.config.get('data_sources', {}):
                    self.config['data_sources']['alpha_vantage']['api_key'] = api_keys['alpha_vantage_api_key']
                    # 如果有API密钥，自动启用数据源
                    self.config['data_sources']['alpha_vantage']['enabled'] = True
                    logger.info(f"Alpha Vantage数据源已启用，API密钥: {api_keys['alpha_vantage_api_key'][:8]}...")
            
            # 更新TuShare Token
            if 'tushare_token' in api_keys and api_keys['tushare_token']:
                if 'tushare' in self.config.get('data_sources', {}):
                    self.config['data_sources']['tushare']['token'] = api_keys['tushare_token']
                    # 如果有Token，自动启用数据源
                    self.config['data_sources']['tushare']['enabled'] = True
                    logger.info(f"TuShare数据源已启用，Token: {api_keys['tushare_token'][:8]}...")
            
            # 更新Polygon API密钥
            if 'polygon_api_key' in api_keys and api_keys['polygon_api_key']:
                if 'polygon' in self.config.get('data_sources', {}):
                    self.config['data_sources']['polygon']['api_key'] = api_keys['polygon_api_key']
                    # 如果有API密钥，自动启用数据源
                    self.config['data_sources']['polygon']['enabled'] = True
                    logger.info(f"Polygon数据源已启用，API密钥: {api_keys['polygon_api_key'][:8]}...")
                    
        except Exception as e:
            logger.error(f"从主配置文件加载API密钥失败: {e}")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "health_check_interval": 300,  # 5分钟
            "retry_config": {
                "max_retries": 3,
                "initial_delay": 1,
                "max_delay": 60,
                "backoff_factor": 2
            },
            "circuit_breaker": {
                "failure_threshold": 3,
                "timeout_minutes": 30,
                "half_open_success_threshold": 2
            },
            "cache": {
                "enabled": True,
                "default_ttl_minutes": 15,
                "max_entries": 1000,
                "cleanup_interval_minutes": 30
            },
            "data_sources": {
                "yahoo_finance": {
                    "enabled": True,
                    "priority": 2,
                    "timeout": 10,
                    "rate_limit": {
                        "calls_per_minute": 200,
                        "calls_per_hour": 2000
                    }
                },
                "alpha_vantage": {
                    "enabled": False,
                    "priority": 1,
                    "api_key": "",  # 需要用户填写
                    "timeout": 10,
                    "rate_limit": {
                        "calls_per_minute": 5,
                        "calls_per_day": 500
                    }
                },
                "tushare": {
                    "enabled": False,
                    "priority": 1,
                    "token": "",  # 需要用户填写
                    "timeout": 10,
                    "rate_limit": {
                        "calls_per_minute": 200,
                        "calls_per_day": 10000
                    }
                },
                "sina_finance": {
                    "enabled": True,
                    "priority": 3,
                    "timeout": 10,
                    "rate_limit": {
                        "calls_per_minute": 100
                    }
                },
                "polygon": {
                    "enabled": False,
                    "priority": 1,
                    "api_key": "",  # 需要用户填写
                    "timeout": 10,
                    "rate_limit": {
                        "calls_per_minute": 5,
                        "calls_per_day": 1000
                    }
                }
            },
            "market_preferences": {
                "domestic": ["tushare", "sina_finance", "yahoo_finance"],
                "international": ["alpha_vantage", "polygon", "yahoo_finance"],
                "crypto": ["alpha_vantage", "polygon"]
            },
            "data_validation": {
                "min_data_points": 1,
                "max_price_change_percent": 50,  # 单日最大涨跌幅
                "required_columns": ["Open", "High", "Low", "Close"]
            },
            "fallback_strategy": {
                "enable_cache": True,
                "cache_duration_hours": 24,
                "enable_partial_data": True,
                "min_success_rate": 0.8
            }
        }
    
    def save_config(self):
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到: {self.config_file}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config.copy()
    
    def get_data_source_config(self, source_name: str) -> Dict[str, Any]:
        """获取特定数据源配置"""
        return self.config.get('data_sources', {}).get(source_name, {})
    
    def update_api_key(self, source_name: str, api_key: str):
        """更新API密钥"""
        if source_name in self.config.get('data_sources', {}):
            if 'api_key' in self.config['data_sources'][source_name]:
                self.config['data_sources'][source_name]['api_key'] = api_key
            elif 'token' in self.config['data_sources'][source_name]:
                self.config['data_sources'][source_name]['token'] = api_key
            self.save_config()
            logger.info(f"已更新 {source_name} 的API密钥")
        else:
            logger.warning(f"未找到数据源: {source_name}")
    
    def enable_data_source(self, source_name: str, enabled: bool = True):
        """启用/禁用数据源"""
        if source_name in self.config.get('data_sources', {}):
            self.config['data_sources'][source_name]['enabled'] = enabled
            self.save_config()
            status = "启用" if enabled else "禁用"
            logger.info(f"已{status}数据源: {source_name}")
        else:
            logger.warning(f"未找到数据源: {source_name}")
    
    def set_priority(self, source_name: str, priority: int):
        """设置数据源优先级"""
        if source_name in self.config.get('data_sources', {}):
            self.config['data_sources'][source_name]['priority'] = priority
            self.save_config()
            logger.info(f"已设置 {source_name} 优先级为: {priority}")
        else:
            logger.warning(f"未找到数据源: {source_name}")
    
    def get_enabled_sources(self) -> Dict[str, Dict[str, Any]]:
        """获取已启用的数据源"""
        enabled_sources = {}
        for name, config in self.config.get('data_sources', {}).items():
            if config.get('enabled', False):
                enabled_sources[name] = config
        return enabled_sources
    
    def validate_config(self) -> Dict[str, list]:
        """验证配置"""
        issues = {
            'errors': [],
            'warnings': []
        }
        
        enabled_sources = self.get_enabled_sources()
        
        # 检查是否有启用的数据源
        if not enabled_sources:
            issues['errors'].append("没有启用的数据源")
        
        # 检查API密钥
        for name, config in enabled_sources.items():
            if name in ['alpha_vantage', 'polygon'] and not config.get('api_key'):
                issues['warnings'].append(f"{name} 已启用但未配置API密钥")
            elif name == 'tushare' and not config.get('token'):
                issues['warnings'].append(f"{name} 已启用但未配置Token")
        
        # 检查优先级设置
        priorities = [config.get('priority', 999) for config in enabled_sources.values()]
        if len(set(priorities)) != len(priorities):
            issues['warnings'].append("存在相同优先级的数据源")
        
        return issues
    
    def create_sample_config(self, file_path: str = None):
        """创建示例配置文件"""
        if file_path is None:
            file_path = os.path.join(os.path.dirname(__file__), 'data_sources_sample.json')
        
        sample_config = self._get_default_config()
        
        # 添加示例API密钥说明
        sample_config['data_sources']['alpha_vantage']['api_key'] = "YOUR_ALPHA_VANTAGE_API_KEY_HERE"
        sample_config['data_sources']['tushare']['token'] = "YOUR_TUSHARE_TOKEN_HERE"
        sample_config['data_sources']['polygon']['api_key'] = "YOUR_POLYGON_API_KEY_HERE"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sample_config, f, indent=2, ensure_ascii=False)
            logger.info(f"示例配置文件已创建: {file_path}")
            logger.info("请根据需要修改配置并重命名为 data_sources.json")
        except Exception as e:
            logger.error(f"创建示例配置文件失败: {e}")

# 环境变量配置支持
class EnvironmentConfig:
    """环境变量配置支持"""
    
    @staticmethod
    def get_api_keys_from_env() -> Dict[str, str]:
        """从环境变量获取API密钥"""
        return {
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY', ''),
            'tushare': os.getenv('TUSHARE_TOKEN', ''),
            'polygon': os.getenv('POLYGON_API_KEY', ''),
            'quandl': os.getenv('QUANDL_API_KEY', ''),
            'iex': os.getenv('IEX_TOKEN', '')
        }
    
    @staticmethod
    def update_config_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
        """使用环境变量更新配置"""
        env_keys = EnvironmentConfig.get_api_keys_from_env()
        
        for source_name, api_key in env_keys.items():
            if api_key and source_name in config.get('data_sources', {}):
                source_config = config['data_sources'][source_name]
                if 'api_key' in source_config:
                    source_config['api_key'] = api_key
                elif 'token' in source_config:
                    source_config['token'] = api_key
                
                # 如果有API密钥，自动启用数据源
                source_config['enabled'] = True
        
        return config

# 便捷函数
def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    config_manager = DataSourceConfig()
    config = config_manager.get_config()
    
    # 尝试从环境变量更新
    config = EnvironmentConfig.update_config_with_env(config)
    
    return config

def create_config_file(file_path: str = None):
    """创建配置文件"""
    config_manager = DataSourceConfig()
    if file_path:
        config_manager.config_file = file_path
    config_manager.save_config()

def setup_api_keys(**kwargs):
    """设置API密钥"""
    config_manager = DataSourceConfig()
    
    for source_name, api_key in kwargs.items():
        if api_key:
            config_manager.update_api_key(source_name, api_key)
            config_manager.enable_data_source(source_name, True)

if __name__ == "__main__":
    # 示例用法
    logger.info("数据源配置管理器")
    logger.info("=" * 50)
    
    # 创建配置管理器
    config_manager = DataSourceConfig()
    
    # 显示当前配置状态
    enabled_sources = config_manager.get_enabled_sources()
    logger.info(f"已启用的数据源: {list(enabled_sources.keys())}")
    
    # 验证配置
    issues = config_manager.validate_config()
    if issues['errors']:
        logger.error(f"配置错误: {issues['errors']}")
    if issues['warnings']:
        logger.warning(f"配置警告: {issues['warnings']}")
    
    # 创建示例配置文件
    config_manager.create_sample_config()
    
    logger.info("配置管理完成！")