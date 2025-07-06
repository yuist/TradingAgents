#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen模型配置管理工具
提供统一的Qwen模型参数配置管理
"""

from typing import Dict, Any, Optional


class QwenConfigManager:
    """Qwen模型配置管理器"""
    
    # 支持思考模式的模型列表
    THINKING_SUPPORTED_MODELS = [
        "qwen3", "qwen-plus", "qwen-turbo", "qwen-max"
    ]
    
    # 支持联网搜索的模型列表
    SEARCH_SUPPORTED_MODELS = [
        "qwen-plus", "qwen-turbo", "qwen-max", "qwen3"
    ]
    
    @classmethod
    def get_extra_body_config(
        cls,
        model: str,
        enable_search: bool = True,
        enable_thinking: Optional[bool] = None,
        forced_search: bool = True,
        search_strategy: str = "pro"
    ) -> Dict[str, Any]:
        """
        获取Qwen模型的extra_body配置
        
        Args:
            model: 模型名称
            enable_search: 是否启用联网搜索
            enable_thinking: 是否启用思考模式（None表示自动判断）
            forced_search: 是否强制搜索
            search_strategy: 搜索策略（"standard" 或 "pro"）
            
        Returns:
            extra_body配置字典
        """
        config = {}
        
        # 配置联网搜索
        if enable_search and cls._is_search_supported(model):
            config["enable_search"] = True
            config["search_options"] = {
                "forced_search": forced_search,
                "search_strategy": search_strategy
            }
        
        # 配置思考模式
        if cls._is_thinking_supported(model):
            if enable_thinking is None:
                # 自动判断：商业版模型默认关闭思考模式以提高响应速度
                enable_thinking = cls._is_open_source_model(model)
            config["enable_thinking"] = enable_thinking
        
        return config
    
    @classmethod
    def get_news_search_config(
        cls,
        model: str,
        task_type: str = "stock_news"
    ) -> Dict[str, Any]:
        """
        获取新闻搜索任务的优化配置
        
        Args:
            model: 模型名称
            task_type: 任务类型（"stock_news" 或 "global_news"）
            
        Returns:
            优化的extra_body配置
        """
        if task_type == "global_news":
            # 全球新闻需要更全面的搜索
            return cls.get_extra_body_config(
                model=model,
                enable_search=True,
                enable_thinking=False,  # 新闻搜索不需要深度思考
                forced_search=True,
                search_strategy="pro"  # 使用pro策略获取更多信息
            )
        else:
            # 股票新闻搜索
            return cls.get_extra_body_config(
                model=model,
                enable_search=True,
                enable_thinking=False,
                forced_search=True,
                search_strategy="standard"  # 标准策略即可满足需求
            )
    
    @classmethod
    def _is_search_supported(cls, model: str) -> bool:
        """检查模型是否支持联网搜索"""
        model_lower = model.lower()
        return any(supported in model_lower for supported in cls.SEARCH_SUPPORTED_MODELS)
    
    @classmethod
    def _is_thinking_supported(cls, model: str) -> bool:
        """检查模型是否支持思考模式"""
        model_lower = model.lower()
        return any(supported in model_lower for supported in cls.THINKING_SUPPORTED_MODELS)
    
    @classmethod
    def _is_open_source_model(cls, model: str) -> bool:
        """判断是否为开源版本模型"""
        model_lower = model.lower()
        # 开源版本通常包含特定标识
        open_source_indicators = ["open", "oss", "free"]
        return any(indicator in model_lower for indicator in open_source_indicators)
    
    @classmethod
    def get_model_capabilities(cls, model: str) -> Dict[str, bool]:
        """获取模型能力信息"""
        return {
            "supports_search": cls._is_search_supported(model),
            "supports_thinking": cls._is_thinking_supported(model),
            "is_open_source": cls._is_open_source_model(model)
        }