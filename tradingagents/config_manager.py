#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingAgents 配置管理模块

该模块负责：
1. 读取和解析 YAML 配置文件
2. 处理环境变量替换
3. 验证配置的完整性
4. 提供配置访问接口
5. 支持多种 LLM 提供商的配置
"""

import os
import yaml
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from dataclasses import dataclass

# 导入统一日志管理器
from .utils.logging_manager import get_logger

logger = get_logger('config', 'manager')


@dataclass
class LLMConfig:
    """LLM 配置数据类"""
    provider: str
    base_url: str
    api_key: str
    models: List[str]
    deep_think_model: str
    quick_think_model: str
    temperature: float = 0.1
    max_tokens: int = 4096
    top_p: float = 0.9


class ConfigManager:
    """
    配置管理器
    
    负责加载、验证和管理 TradingAgents 的所有配置
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为 None 则使用默认路径
        """
        self.config_path = config_path or self._find_config_file()
        self.config = {}
        self._load_config()
    
    def _find_config_file(self) -> str:
        """
        查找配置文件
        
        按优先级查找配置文件：
        1. 环境变量 TRADINGAGENTS_CONFIG
        2. 当前目录的 config.yaml
        3. 项目根目录的 config.yaml
        """
        # 1. 检查环境变量
        env_config = os.getenv('TRADINGAGENTS_CONFIG')
        if env_config and os.path.exists(env_config):
            return env_config
        
        # 2. 检查当前目录
        current_dir_config = Path('./config.yaml')
        if current_dir_config.exists():
            return str(current_dir_config)
        
        # 3. 检查项目根目录
        project_root = Path(__file__).parent.parent
        root_config = project_root / 'config.yaml'
        if root_config.exists():
            return str(root_config)
        
        # 如果都没找到，返回默认路径
        return str(current_dir_config)
    
    def _load_config(self):
        """
        加载配置文件
        """
        try:
            if not os.path.exists(self.config_path):
                logger.warning(
                    "Configuration file not found, using default config",
                    extra={
                        'component': 'config_manager',
                        'config_path': self.config_path,
                        'function': '_load_config'
                    }
                )
                self._create_default_config()
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            # 处理环境变量替换
            self.config = self._resolve_env_vars(raw_config)
            
            # 验证配置
            self._validate_config()
            
            logger.info(
                "Configuration file loaded successfully",
                extra={
                    'component': 'config_manager',
                    'config_path': self.config_path,
                    'function': '_load_config'
                }
            )
            
        except Exception as e:
            logger.error(
                "Failed to load configuration file",
                extra={
                    'component': 'config_manager',
                    'config_path': self.config_path,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'function': '_load_config'
                }
            )
            raise
    
    def _resolve_env_vars(self, obj: Any) -> Any:
        """
        递归解析环境变量
        
        支持格式：
        - ${VAR_NAME}
        - ${VAR_NAME:default_value}
        """
        if isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # 匹配 ${VAR_NAME} 或 ${VAR_NAME:default} 格式
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replace_env_var(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ''
                return os.getenv(var_name, default_value)
            
            return re.sub(pattern, replace_env_var, obj)
        else:
            return obj
    
    def _validate_config(self):
        """
        验证配置的完整性
        """
        try:
            # 验证必需的顶级部分
            required_sections = ['api_keys', 'llm_providers', 'system', 'models']
            
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"配置文件缺少必需的部分: {section}")
            
            # 验证system配置
            self._validate_system_config()
            
            # 验证models配置
            self._validate_models_config()
            
            # 验证LLM提供商配置
            self._validate_llm_providers_config()
            
            # 验证API密钥配置
            self._validate_api_keys_config()
            
            # 验证日志配置（可选）
            if 'logging' in self.config:
                self._validate_logging_config()
            
            logger.info(
                "Configuration validation passed",
                extra={'component': 'config_manager', 'function': '_validate_config'}
            )
            
        except Exception as e:
            logger.error(
                "Configuration validation failed",
                extra={
                    'component': 'config_manager',
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'function': '_validate_config'
                }
            )
            raise
    
    def _validate_system_config(self):
        """验证系统配置"""
        system_config = self.config.get('system', {})
        
        # 验证必需的系统配置项
        required_system_keys = ['project_dir', 'data_dir']
        for key in required_system_keys:
            if key not in system_config:
                logger.warning(f"系统配置缺少 {key}，使用默认值")
                system_config[key] = self._get_default_system_value(key)
        
        # 验证目录路径
        for dir_key in ['project_dir', 'data_dir', 'results_dir']:
            if dir_key in system_config:
                dir_path = system_config[dir_key]
                if not isinstance(dir_path, str):
                    raise ValueError(f"系统配置 {dir_key} 必须是字符串")
                
                # 确保目录存在
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except OSError as e:
                    logger.warning(f"无法创建目录 {dir_path}: {e}")
        
        # 验证数值配置
        if 'max_recur_limit' in system_config:
            max_recur = system_config['max_recur_limit']
            if not isinstance(max_recur, int) or max_recur <= 0:
                logger.warning("max_recur_limit 必须是正整数，使用默认值 100")
                system_config['max_recur_limit'] = 100
    
    def _validate_models_config(self):
        """验证模型配置"""
        models_config = self.config.get('models', {})
        
        # 验证LLM提供商
        if 'llm_provider' not in models_config:
            raise ValueError("models配置缺少 llm_provider")
        
        current_provider = models_config['llm_provider']
        if current_provider not in self.config['llm_providers']:
            raise ValueError(f"未知的 LLM 提供商: {current_provider}")
        
        # 验证模型名称
        provider_config = self.config['llm_providers'][current_provider]
        models_config_data = provider_config.get('models', [])
        
        # 获取所有可用模型列表
        available_models = []
        if isinstance(models_config_data, list):
            # 简单列表格式
            available_models = models_config_data
        elif isinstance(models_config_data, dict):
            # 嵌套字典格式，收集所有子列表中的模型
            for category_models in models_config_data.values():
                if isinstance(category_models, list):
                    available_models.extend(category_models)
        
        for model_key in ['deep_think_llm', 'quick_think_llm']:
            if model_key in models_config:
                model_name = models_config[model_key]
                if model_name not in available_models:
                    logger.warning(f"模型 {model_name} 不在提供商 {current_provider} 的可用模型列表中")
                    logger.warning(f"可用模型列表: {available_models}")
        
        # 验证数值参数
        numeric_params = {
            'temperature': (0.0, 2.0),
            'max_tokens': (1, 100000),
            'top_p': (0.0, 1.0)
        }
        
        for param, (min_val, max_val) in numeric_params.items():
            if param in models_config:
                value = models_config[param]
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    logger.warning(f"参数 {param} 值 {value} 超出范围 [{min_val}, {max_val}]，使用默认值")
                    models_config[param] = self._get_default_model_value(param)
    
    def _validate_llm_providers_config(self):
        """验证LLM提供商配置"""
        providers_config = self.config.get('llm_providers', {})
        
        for provider_name, provider_config in providers_config.items():
            if not isinstance(provider_config, dict):
                raise ValueError(f"LLM提供商 {provider_name} 的配置必须是字典")
            
            # 验证必需的配置项
            required_keys = ['base_url', 'models']
            for key in required_keys:
                if key not in provider_config:
                    raise ValueError(f"LLM提供商 {provider_name} 缺少必需的配置项: {key}")
            
            # 验证base_url格式
            base_url = provider_config['base_url']
            if not isinstance(base_url, str) or not base_url.startswith(('http://', 'https://')):
                raise ValueError(f"LLM提供商 {provider_name} 的 base_url 格式无效: {base_url}")
            
            # 验证models配置 - 支持两种格式：简单列表或嵌套字典
            models = provider_config['models']
            if isinstance(models, list):
                # 简单列表格式
                if len(models) == 0:
                    raise ValueError(f"LLM提供商 {provider_name} 的 models 列表不能为空")
            elif isinstance(models, dict):
                # 嵌套字典格式 (deep/quick)
                if not models:
                    raise ValueError(f"LLM提供商 {provider_name} 的 models 字典不能为空")
                
                # 验证每个子类别都是非空列表
                for category, model_list in models.items():
                    if not isinstance(model_list, list) or len(model_list) == 0:
                        raise ValueError(f"LLM提供商 {provider_name} 的 models.{category} 必须是非空列表")
            else:
                raise ValueError(f"LLM提供商 {provider_name} 的 models 必须是列表或字典格式")
    
    def _validate_api_keys_config(self):
        """验证API密钥配置"""
        api_keys_config = self.config.get('api_keys', {})
        
        # 检查当前使用的提供商是否有API密钥
        current_provider = self.config['models']['llm_provider']
        
        # 首先尝试直接使用提供商名称作为键
        api_key = api_keys_config.get(current_provider, '')
        
        # 如果没找到，尝试使用 provider_api_key 格式
        if not api_key:
            api_key_name = f"{current_provider}_api_key"
            api_key = api_keys_config.get(api_key_name, '')
        
        if not api_key:
            logger.warning(f"当前LLM提供商 {current_provider} 的API密钥未配置或为空")
    
    def _validate_logging_config(self):
        """验证日志配置"""
        logging_config = self.config.get('logging', {})
        
        # 验证基本配置
        if 'log_dir' in logging_config:
            log_dir = logging_config['log_dir']
            if not isinstance(log_dir, str):
                raise ValueError("logging.log_dir 必须是字符串")
        
        # 验证日志级别
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if 'root_level' in logging_config:
            level = logging_config['root_level']
            if level not in valid_levels:
                raise ValueError(f"logging.root_level 必须是以下值之一: {valid_levels}")
        
        # 验证数值配置
        numeric_configs = {
            'max_file_size': (1024, 1024*1024*1024),  # 1KB to 1GB
            'backup_count': (1, 100),
            'retention_days': (1, 365)
        }
        
        for config_key, (min_val, max_val) in numeric_configs.items():
            if config_key in logging_config:
                value = logging_config[config_key]
                if not isinstance(value, int) or not (min_val <= value <= max_val):
                    logger.warning(f"日志配置 {config_key} 值 {value} 超出范围 [{min_val}, {max_val}]")
        
        # 验证分类配置
        if 'categories' in logging_config:
            categories = logging_config['categories']
            if not isinstance(categories, dict):
                raise ValueError("logging.categories 必须是字典")
            
            for category_name, category_config in categories.items():
                if not isinstance(category_config, dict):
                    raise ValueError(f"日志分类 {category_name} 的配置必须是字典")
                
                # 验证级别
                if 'level' in category_config:
                    level = category_config['level']
                    if level not in valid_levels:
                        raise ValueError(f"日志分类 {category_name} 的级别必须是以下值之一: {valid_levels}")
                
                # 验证文件名
                if 'file' in category_config:
                    filename = category_config['file']
                    if not isinstance(filename, str) or not filename.endswith('.log'):
                        logger.warning(f"日志分类 {category_name} 的文件名建议以 .log 结尾")
        
        # 验证环境特定配置
        for env in ['development', 'production']:
            if env in logging_config:
                env_config = logging_config[env]
                if not isinstance(env_config, dict):
                    raise ValueError(f"logging.{env} 必须是字典")
                
                # 验证环境特定的级别设置
                if 'root_level' in env_config:
                    level = env_config['root_level']
                    if level not in valid_levels:
                        raise ValueError(f"logging.{env}.root_level 必须是以下值之一: {valid_levels}")
        
        logger.info("日志配置验证通过")
    
    def _get_default_system_value(self, key: str) -> Any:
        """获取系统配置的默认值"""
        defaults = {
            'project_dir': '.',
            'data_dir': './data',
            'results_dir': './results',
            'data_cache_dir': './cache',
            'max_recur_limit': 100,
            'online_tools': True
        }
        return defaults.get(key, '')
    
    def _get_default_model_value(self, key: str) -> Any:
        """获取模型配置的默认值"""
        defaults = {
            'temperature': 0.1,
            'max_tokens': 4096,
            'top_p': 0.9
        }
        return defaults.get(key, 0)
    
    def _create_default_config(self):
        """
        创建默认配置
        """
        from .default_config import DEFAULT_CONFIG
        
        # 转换旧配置格式到新格式
        self.config = {
            'api_keys': {
                'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
                'finnhub_api_key': os.getenv('FINNHUB_API_KEY', ''),
                'alpha_vantage_api_key': os.getenv('ALPHA_VANTAGE_API_KEY', ''),
                'polygon_api_key': os.getenv('POLYGON_API_KEY', ''),
                'google_api_key': os.getenv('GOOGLE_API_KEY', ''),
                'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY', ''),
                'openrouter_api_key': os.getenv('OPENROUTER_API_KEY', ''),
                'deepseek_api_key': os.getenv('DEEPSEEK_API_KEY', ''),
                'qwen_api_key': os.getenv('QWEN_API_KEY', ''),
            },
            'llm_providers': {
                'openai': {
                    'base_url': 'https://api.openai.com/v1',
                    'models': ['gpt-4o', 'gpt-4o-mini', 'o1-preview', 'o1-mini']
                },
                'google': {
                    'base_url': 'https://generativelanguage.googleapis.com/v1',
                    'models': ['gemini-2.0-flash', 'gemini-1.5-pro']
                },
                'anthropic': {
                    'base_url': 'https://api.anthropic.com',
                    'models': ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022']
                }
            },
            'system': {
                'project_dir': DEFAULT_CONFIG.get('project_dir', '.'),
                'results_dir': DEFAULT_CONFIG.get('results_dir', './results'),
                'data_dir': DEFAULT_CONFIG.get('data_dir', './data'),
                'data_cache_dir': DEFAULT_CONFIG.get('data_cache_dir', './cache'),
                'max_recur_limit': DEFAULT_CONFIG.get('max_recur_limit', 100),
                'online_tools': DEFAULT_CONFIG.get('online_tools', True)
            },
            'models': {
                'llm_provider': DEFAULT_CONFIG.get('llm_provider', 'openai'),
                'deep_think_llm': DEFAULT_CONFIG.get('deep_think_llm', 'gpt-4o-mini'),
                'quick_think_llm': DEFAULT_CONFIG.get('quick_think_llm', 'gpt-4o-mini'),
                'temperature': 0.1,
                'max_tokens': 4096,
                'top_p': 0.9
            },
            'debate': {
                'max_debate_rounds': DEFAULT_CONFIG.get('max_debate_rounds', 1),
                'max_risk_discuss_rounds': DEFAULT_CONFIG.get('max_risk_discuss_rounds', 1)
            }
        }
    
    def get_llm_config(self) -> LLMConfig:
        """
        获取当前 LLM 配置
        
        Returns:
            LLMConfig: LLM 配置对象
        """
        provider = self.config['models']['llm_provider']
        provider_config = self.config['llm_providers'][provider]
        
        # 获取 API 密钥
        api_key_map = {
            'openai': 'openai',
            'google': 'google',
            'anthropic': 'anthropic',
            'openrouter': 'openrouter',
            'deepseek': 'deepseek',
            'qwen': 'qwen',
            'ollama': ''  # 本地部署不需要 API 密钥
        }
        
        api_key_name = api_key_map.get(provider, f'{provider}_api_key')
        api_key = self.config['api_keys'].get(api_key_name, '') if api_key_name else ''
        
        # 处理models格式兼容性 - 转换为LLMConfig期望的列表格式
        models_config_data = provider_config['models']
        models_list = []
        if isinstance(models_config_data, list):
            # 简单列表格式
            models_list = models_config_data
        elif isinstance(models_config_data, dict):
            # 嵌套字典格式，收集所有子列表中的模型
            for category_models in models_config_data.values():
                if isinstance(category_models, list):
                    models_list.extend(category_models)
        
        return LLMConfig(
            provider=provider,
            base_url=provider_config['base_url'],
            api_key=api_key,
            models=models_list,
            deep_think_model=self.config['models']['deep_think_llm'],
            quick_think_model=self.config['models']['quick_think_llm'],
            temperature=self.config['models'].get('temperature', 0.1),
            max_tokens=self.config['models'].get('max_tokens', 4096),
            top_p=self.config['models'].get('top_p', 0.9)
        )
    
    def get_api_key(self, service: str) -> str:
        """
        获取指定服务的 API 密钥
        
        Args:
            service: 服务名称 (如 'openai', 'anthropic' 等)
        
        Returns:
            str: API 密钥
        """
        # 首先尝试直接使用服务名作为键
        api_key = self.config['api_keys'].get(service, '')
        if api_key:
            return api_key
        
        # 如果没找到，尝试使用 service_api_key 格式
        key_name = f'{service}_api_key'
        return self.config['api_keys'].get(key_name, '')
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """
        获取嵌入模型配置
        
        Returns:
            Dict: 嵌入模型配置字典，包含provider, model, api_key, base_url, dimensions
        """
        # 检查是否有专门的嵌入配置
        if 'embedding' in self.config.get('models', {}):
            embedding_config = self.config['models']['embedding']
            
            # 获取API密钥
            api_key_source = embedding_config.get('api_key_source', 'openai')
            api_key = self.get_api_key(api_key_source)
            
            return {
                'provider': embedding_config.get('provider', 'openai'),
                'model': embedding_config.get('model', 'text-embedding-3-small'),
                'api_key': api_key,
                'base_url': embedding_config.get('base_url', 'https://api.openai.com/v1'),
                'dimensions': embedding_config.get('dimensions', 1536)
            }
        
        # 如果没有专门的嵌入配置，使用默认的OpenAI配置
        return {
            'provider': 'openai',
            'model': 'text-embedding-3-small',
            'api_key': self.get_api_key('openai'),
            'base_url': 'https://api.openai.com/v1',
            'dimensions': 1536
        }
    
    def get_system_config(self) -> Dict[str, Any]:
        """
        获取系统配置
        
        Returns:
            Dict: 系统配置字典
        """
        return self.config['system'].copy()
    
    def get_debate_config(self) -> Dict[str, Any]:
        """
        获取辩论配置
        
        Returns:
            Dict: 辩论配置字典
        """
        return self.config['debate'].copy()
    
    def get_logging_config(self, environment: str = None) -> Dict[str, Any]:
        """
        获取日志配置
        
        Args:
            environment: 环境名称 ('development' 或 'production')，如果未指定则从环境变量获取
        
        Returns:
            Dict: 日志配置字典
        """
        # 确定当前环境
        if environment is None:
            environment = os.getenv('TRADING_ENV', 'development')
        
        default_logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'root_level': 'INFO',
            'log_dir': 'logs',
            'max_file_size': 52428800,  # 50MB
            'backup_count': 10,
            'retention_days': 30,
            'enable_async': True,
            'enable_structured': True,
            'enable_console': False,  # 默认关闭控制台输出
            'categories': {
                'trading': {'level': 'INFO', 'file': 'trading.log'},
                'dataflow': {'level': 'INFO', 'file': 'dataflow.log'},
                'agents': {'level': 'INFO', 'file': 'agents.log'},
                'cli': {'level': 'INFO', 'file': 'cli.log'},
                'system': {'level': 'WARNING', 'file': 'system.log'},
                'performance': {'level': 'INFO', 'file': 'performance.log'}
            }
        }
        
        # 如果配置文件中有日志配置，则合并
        if 'logging' in self.config:
            user_config = self.config['logging'].copy()
            
            # 应用环境特定配置
            if environment in user_config:
                env_config = user_config[environment]
                # 移除环境配置，避免重复合并
                user_config.pop(environment, None)
                user_config.pop('development', None)
                user_config.pop('production', None)
                # 合并环境特定配置
                user_config.update(env_config)
            
            # 深度合并categories配置
            if 'categories' in user_config and 'categories' in default_logging_config:
                default_categories = default_logging_config['categories'].copy()
                # 深度合并每个分类的配置，确保file字段不丢失
                for category, category_config in user_config['categories'].items():
                    if category in default_categories:
                        # 合并现有分类的配置，保留默认的file字段
                        merged_config = default_categories[category].copy()
                        merged_config.update(category_config)
                        default_categories[category] = merged_config
                    else:
                        # 新分类，确保有file字段
                        if 'file' not in category_config:
                            category_config['file'] = f'{category}.log'
                        default_categories[category] = category_config
                user_config['categories'] = default_categories
            
            default_logging_config.update(user_config)
        
        return default_logging_config
    
    def get_recommended_config(self, config_name: str) -> Dict[str, Any]:
        """
        获取推荐配置
        
        Args:
            config_name: 配置名称 (如 'high_performance', 'balanced' 等)
        
        Returns:
            Dict: 推荐配置字典
        """
        if 'recommended_configs' not in self.config:
            return {}
        
        return self.config['recommended_configs'].get(config_name, {})
    
    def apply_recommended_config(self, config_name: str):
        """
        应用推荐配置
        
        Args:
            config_name: 配置名称
        """
        recommended = self.get_recommended_config(config_name)
        if not recommended:
            logger.warning(f"未找到推荐配置: {config_name}")
            return
        
        # 更新模型配置
        for key, value in recommended.items():
            if key in self.config['models']:
                self.config['models'][key] = value
            elif key in self.config['debate']:
                self.config['debate'][key] = value
        
        logger.info(f"已应用推荐配置: {config_name}")
    
    def to_legacy_config(self) -> Dict[str, Any]:
        """
        转换为旧版配置格式，保持向后兼容
        
        Returns:
            Dict: 旧版配置格式
        """
        llm_config = self.get_llm_config()
        system_config = self.get_system_config()
        debate_config = self.get_debate_config()
        
        return {
            'project_dir': system_config['project_dir'],
            'results_dir': system_config['results_dir'],
            'data_dir': system_config['data_dir'],
            'data_cache_dir': system_config['data_cache_dir'],
            'llm_provider': llm_config.provider,
            'deep_think_llm': llm_config.deep_think_model,
            'quick_think_llm': llm_config.quick_think_model,
            'backend_url': llm_config.base_url,
            'max_debate_rounds': debate_config['max_debate_rounds'],
            'max_risk_discuss_rounds': debate_config['max_risk_discuss_rounds'],
            'max_recur_limit': system_config['max_recur_limit'],
            'online_tools': system_config['online_tools']
        }
    
    def list_available_models(self, provider: Optional[str] = None) -> Dict[str, List[str]]:
        """
        列出可用的模型
        
        Args:
            provider: 指定提供商，如果为 None 则返回所有提供商的模型
        
        Returns:
            Dict: 提供商到模型列表的映射
        """
        if provider:
            if provider in self.config['llm_providers']:
                return {provider: self.config['llm_providers'][provider]['models']}
            else:
                return {}
        
        return {
            p: config['models'] 
            for p, config in self.config['llm_providers'].items()
        }
    
    def validate_model(self, provider: str, model: str) -> bool:
        """
        验证模型是否可用
        
        Args:
            provider: 提供商名称
            model: 模型名称
        
        Returns:
            bool: 模型是否可用
        """
        if provider not in self.config['llm_providers']:
            return False
        
        models_config = self.config['llm_providers'][provider]['models']
        
        # 处理嵌套结构（deep/quick分类）
        if isinstance(models_config, dict):
            all_models = []
            for category_models in models_config.values():
                if isinstance(category_models, list):
                    all_models.extend(category_models)
            return model in all_models
        
        # 处理平坦列表结构
        elif isinstance(models_config, list):
            return model in models_config
        
        return False


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Returns:
        ConfigManager: 配置管理器实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


    @classmethod
    def from_legacy_config(cls, legacy_config: dict) -> 'ConfigManager':
        """
        从传统配置字典创建配置管理器（向后兼容）
        
        Args:
            legacy_config: 传统配置字典
            
        Returns:
            ConfigManager: 配置管理器实例
        """
        # 转换传统配置为新格式
        new_config = {
            'api_keys': {},
            'llm': {
                'provider': legacy_config.get('llm_provider', 'openai'),
                'deep_think_model': legacy_config.get('deep_think_llm', 'gpt-4'),
                'quick_think_model': legacy_config.get('quick_think_llm', 'gpt-3.5-turbo'),
                'base_url': legacy_config.get('backend_url', ''),
                'temperature': 0.7,
                'max_tokens': 4000,
                'top_p': 1.0
            },
            'system': {
                'project_dir': legacy_config.get('project_dir', './'),
                'results_dir': legacy_config.get('results_dir', './results'),
                'data_dir': legacy_config.get('data_dir', './data'),
                'use_online_tools': legacy_config.get('use_online_tools', True)
            },
            'debate': {
                'max_rounds': legacy_config.get('max_debate_rounds', 3),
                'max_recursion_limit': legacy_config.get('max_recursion_limit', 10)
            }
        }
        
        instance = cls.__new__(cls)
        instance.config = new_config
        instance._validate_config()
        return instance


def reload_config(config_path: Optional[str] = None):
    """
    重新加载配置
    
    Args:
        config_path: 新的配置文件路径
    """
    global _config_manager
    _config_manager = ConfigManager(config_path)