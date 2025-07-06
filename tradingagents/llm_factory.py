#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingAgents LLM 工厂模块

该模块负责：
1. 创建和配置各种 LLM 提供商的客户端
2. 统一不同提供商的接口
3. 处理 API 密钥和认证
4. 提供模型验证和错误处理
"""

import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

# LangChain LLM 导入
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

from .config_manager import ConfigManager, LLMConfig
from .utils.logging_manager import get_logger

# 创建日志器
logger = get_logger('llm', 'factory')


@dataclass
class LLMInstance:
    """LLM 实例包装器"""
    model: BaseChatModel
    provider: str
    model_name: str
    config: Dict[str, Any]


class LLMFactory:
    """
    LLM 工厂类
    
    负责创建和管理各种 LLM 提供商的实例
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        初始化 LLM 工厂
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager
        self._instances_cache = {}  # 实例缓存
    
    def create_llm(
        self, 
        provider: str, 
        model_name: str, 
        **kwargs
    ) -> BaseChatModel:
        """
        创建 LLM 实例
        
        Args:
            provider: 提供商名称
            model_name: 模型名称
            **kwargs: 额外的模型参数
        
        Returns:
            BaseChatModel: LLM 实例
        """
        # 生成缓存键
        cache_key = f"{provider}:{model_name}:{hash(str(sorted(kwargs.items())))}"
        
        # 检查缓存
        if cache_key in self._instances_cache:
            return self._instances_cache[cache_key]
        
        # 获取配置
        if self.config_manager:
            llm_config = self.config_manager.get_llm_config()
            provider_config = self.config_manager.config['llm_providers'].get(provider, {})
        else:
            # 使用默认配置
            llm_config = None
            provider_config = {}
        
        # 创建实例
        instance = self._create_instance(provider, model_name, provider_config, llm_config, **kwargs)
        
        # 缓存实例
        self._instances_cache[cache_key] = instance
        
        return instance
    
    def _create_instance(
        self, 
        provider: str, 
        model_name: str, 
        provider_config: Dict[str, Any],
        llm_config: Optional[LLMConfig],
        **kwargs
    ) -> BaseChatModel:
        """
        创建具体的 LLM 实例
        
        Args:
            provider: 提供商名称
            model_name: 模型名称
            provider_config: 提供商配置
            llm_config: LLM 配置
            **kwargs: 额外参数
        
        Returns:
            BaseChatModel: LLM 实例
        """
        # 构建基础配置，优先使用kwargs，然后是llm_config，最后是默认值
        config = {
            'model': model_name,
            'temperature': kwargs.get('temperature') or (llm_config.temperature if llm_config else 0.1),
            'max_tokens': kwargs.get('max_tokens') or (llm_config.max_tokens if llm_config else 4096),
            'top_p': kwargs.get('top_p') or (llm_config.top_p if llm_config else 0.9),
            # 添加超时和重试配置
            'timeout': kwargs.get('timeout', 120),  # 2分钟超时
            'max_retries': kwargs.get('max_retries', 3),  # 最多重试3次
            'request_timeout': kwargs.get('request_timeout', 60),  # 请求超时60秒
        }
        
        # 从kwargs中移除已经处理的参数，避免重复传递
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens', 'top_p']}
        
        # 根据提供商创建实例
        if provider.lower() == 'openai':
            return self._create_openai_instance(provider_config, llm_config, config, **filtered_kwargs)
        elif provider.lower() == 'anthropic':
            return self._create_anthropic_instance(provider_config, llm_config, config, **filtered_kwargs)
        elif provider.lower() == 'google':
            return self._create_google_instance(provider_config, llm_config, config, **filtered_kwargs)
        elif provider.lower() == 'openrouter':
            return self._create_openrouter_instance(provider_config, llm_config, config, **filtered_kwargs)
        elif provider.lower() == 'deepseek':
            return self._create_deepseek_instance(provider_config, llm_config, config, **filtered_kwargs)
        elif provider.lower() == 'qwen':
            return self._create_qwen_instance(provider_config, llm_config, config, **filtered_kwargs)
        elif provider.lower() == 'ollama':
            return self._create_ollama_instance(provider_config, llm_config, config, **filtered_kwargs)
        else:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")
    
    def _create_openai_instance(
        self, 
        provider_config: Dict[str, Any],
        llm_config: Optional[LLMConfig],
        config: Dict[str, Any],
        **kwargs
    ) -> ChatOpenAI:
        """
        创建 OpenAI 实例
        """
        # 获取 API 密钥
        api_key = self._get_api_key('openai', llm_config)
        
        # 获取基础 URL
        base_url = provider_config.get('base_url', 'https://api.openai.com/v1')
        
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            **config,
            **kwargs
        )
    
    def _create_anthropic_instance(
        self, 
        provider_config: Dict[str, Any],
        llm_config: Optional[LLMConfig],
        config: Dict[str, Any],
        **kwargs
    ) -> ChatAnthropic:
        """
        创建 Anthropic 实例
        """
        # 获取 API 密钥
        api_key = self._get_api_key('anthropic', llm_config)
        
        # 获取基础 URL
        base_url = provider_config.get('base_url')
        
        anthropic_config = config.copy()
        if base_url:
            anthropic_config['base_url'] = base_url
        
        return ChatAnthropic(
            api_key=api_key,
            **anthropic_config,
            **kwargs
        )
    
    def _create_google_instance(
        self, 
        provider_config: Dict[str, Any],
        llm_config: Optional[LLMConfig],
        config: Dict[str, Any],
        **kwargs
    ) -> ChatGoogleGenerativeAI:
        """
        创建 Google 实例
        """
        # 获取 API 密钥
        api_key = self._get_api_key('google', llm_config)
        
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            **config,
            **kwargs
        )
    
    def _create_openrouter_instance(
        self, 
        provider_config: Dict[str, Any],
        llm_config: Optional[LLMConfig],
        config: Dict[str, Any],
        **kwargs
    ) -> ChatOpenAI:
        """
        创建 OpenRouter 实例 (使用 OpenAI 兼容接口)
        """
        # 获取 API 密钥
        api_key = self._get_api_key('openrouter', llm_config)
        
        # OpenRouter 使用 OpenAI 兼容接口
        base_url = provider_config.get('base_url', 'https://openrouter.ai/api/v1')
        
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            **config,
            **kwargs
        )
    
    def _create_deepseek_instance(
        self, 
        provider_config: Dict[str, Any],
        llm_config: Optional[LLMConfig],
        config: Dict[str, Any],
        **kwargs
    ) -> ChatOpenAI:
        """
        创建 DeepSeek 实例 (使用 OpenAI 兼容接口)
        """
        # 获取 API 密钥
        api_key = self._get_api_key('deepseek', llm_config)
        
        # DeepSeek 使用 OpenAI 兼容接口
        base_url = provider_config.get('base_url', 'https://api.deepseek.com/v1')
        
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            **config,
            **kwargs
        )
    
    def _create_qwen_instance(
        self, 
        provider_config: Dict[str, Any],
        llm_config: Optional[LLMConfig],
        config: Dict[str, Any],
        **kwargs
    ) -> ChatOpenAI:
        """
        创建 Qwen 实例 (使用 OpenAI 兼容接口)
        """
        # 获取 API 密钥
        api_key = self._get_api_key('qwen', llm_config)
        
        # Qwen 使用 OpenAI 兼容接口
        base_url = provider_config.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            **config,
            **kwargs
        )
    
    def _create_ollama_instance(
        self, 
        provider_config: Dict[str, Any],
        llm_config: Optional[LLMConfig],
        config: Dict[str, Any],
        **kwargs
    ) -> ChatOpenAI:
        """
        创建 Ollama 实例 (本地部署，使用 OpenAI 兼容接口)
        """
        # Ollama 不需要 API 密钥
        base_url = provider_config.get('base_url', 'http://localhost:11434/v1')
        
        return ChatOpenAI(
            api_key='ollama',  # Ollama 需要一个非空的 API 密钥
            base_url=base_url,
            **config,
            **kwargs
        )
    
    def _get_api_key(self, provider: str, llm_config: Optional[LLMConfig]) -> str:
        """
        获取 API 密钥
        
        Args:
            provider: 提供商名称
            llm_config: LLM 配置
        
        Returns:
            str: API 密钥
        """
        # 优先使用配置管理器中的密钥
        if self.config_manager:
            api_key = self.config_manager.get_api_key(provider)
            if api_key:
                return api_key
        
        # 使用 LLM 配置中的密钥
        if llm_config and llm_config.api_key:
            return llm_config.api_key
        
        # 从环境变量获取
        env_key_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY',
            'deepseek': 'DEEPSEEK_API_KEY',
            'qwen': 'QWEN_API_KEY',
        }
        
        env_key = env_key_map.get(provider)
        if env_key:
            api_key = os.getenv(env_key)
            if api_key:
                return api_key
        
        # 如果是 Ollama，返回默认值
        if provider == 'ollama':
            return 'ollama'
        
        # 抛出错误
        raise ValueError(f"未找到 {provider} 的 API 密钥")
    
    def validate_model(self, provider: str, model_name: str) -> bool:
        """
        验证模型是否可用
        
        Args:
            provider: 提供商名称
            model_name: 模型名称
        
        Returns:
            bool: 模型是否可用
        """
        if self.config_manager:
            return self.config_manager.validate_model(provider, model_name)
        
        # 如果没有配置管理器，假设模型可用
        return True
    
    def list_available_models(self, provider: Optional[str] = None) -> Dict[str, list]:
        """
        列出可用模型
        
        Args:
            provider: 提供商名称，如果为 None 则返回所有提供商
        
        Returns:
            Dict: 提供商到模型列表的映射
        """
        if self.config_manager:
            return self.config_manager.list_available_models(provider)
        
        # 返回默认模型列表
        default_models = {
            'openai': ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
            'anthropic': ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022'],
            'google': ['gemini-2.0-flash', 'gemini-1.5-pro'],
        }
        
        if provider:
            return {provider: default_models.get(provider, [])}
        
        return default_models
    
    def clear_cache(self):
        """
        清除实例缓存
        """
        self._instances_cache.clear()
        logger.info("LLM 实例缓存已清除")


# 全局工厂实例
_llm_factory: Optional[LLMFactory] = None


def get_llm_factory(config_manager: Optional[ConfigManager] = None) -> LLMFactory:
    """
    获取全局 LLM 工厂实例
    
    Args:
        config_manager: 配置管理器实例
    
    Returns:
        LLMFactory: LLM 工厂实例
    """
    global _llm_factory
    if _llm_factory is None or (config_manager is not None and _llm_factory.config_manager != config_manager):
        _llm_factory = LLMFactory(config_manager)
    return _llm_factory


def create_llm(
    provider: str, 
    model_name: str, 
    config_manager: Optional[ConfigManager] = None,
    **kwargs
) -> BaseChatModel:
    """
    便捷函数：创建 LLM 实例
    
    Args:
        provider: 提供商名称
        model_name: 模型名称
        config_manager: 配置管理器实例
        **kwargs: 额外参数
    
    Returns:
        BaseChatModel: LLM 实例
    """
    factory = get_llm_factory(config_manager)
    return factory.create_llm(provider, model_name, **kwargs)