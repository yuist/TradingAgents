#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen 大模型使用示例

此示例展示如何在 TradingAgents 框架中使用 Qwen 大模型进行金融分析。
"""

import os
import sys
from typing import Dict, Any

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.config_manager import get_config_manager
from tradingagents.llm_factory import create_llm
from langchain.schema import HumanMessage


def test_qwen_basic_chat():
    """
    测试 Qwen 模型的基本对话功能
    """
    print("=== Qwen 基本对话测试 ===")
    
    try:
        # 创建 Qwen 模型实例
        config_manager = get_config_manager()
        qwen_llm = create_llm(
            provider='qwen',
            model_name='qwen-turbo',
            config_manager=config_manager
        )
        
        # 测试基本对话
        message = HumanMessage(content="你好，请简单介绍一下你自己。")
        response = qwen_llm.invoke([message])
        
        print(f"✅ Qwen 响应: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Qwen 基本对话测试失败: {e}")
        return False


def test_qwen_financial_analysis():
    """
    测试 Qwen 模型的金融分析能力
    """
    print("\n=== Qwen 金融分析测试 ===")
    
    try:
        # 创建 Qwen 模型实例
        config_manager = get_config_manager()
        qwen_llm = create_llm(
            provider='qwen',
            model_name='qwen-plus',
            config_manager=config_manager
        )
        
        # 金融分析提示
        financial_prompt = """
        请分析以下股票信息并给出投资建议：
        
        公司：苹果公司 (AAPL)
        当前股价：$150
        市盈率：25
        市净率：5.2
        近期新闻：发布了新款iPhone，销量预期良好
        
        请从技术面、基本面和市场情绪三个角度进行分析，并给出明确的投资建议。
        """
        
        message = HumanMessage(content=financial_prompt)
        response = qwen_llm.invoke([message])
        
        print(f"✅ Qwen 金融分析结果:")
        print(f"{response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Qwen 金融分析测试失败: {e}")
        return False


def test_qwen_different_models():
    """
    测试不同的 Qwen 模型
    """
    print("\n=== Qwen 不同模型测试 ===")
    
    models_to_test = [
        ('qwen-turbo', '快速模型'),
        ('qwen-plus', '平衡模型'),
        ('qwen-max', '高级模型')
    ]
    
    config_manager = get_config_manager()
    
    for model_name, description in models_to_test:
        try:
            print(f"\n测试 {model_name} ({description})...")
            
            qwen_llm = create_llm(
                provider='qwen',
                model_name=model_name,
                config_manager=config_manager
            )
            
            message = HumanMessage(content="请用一句话描述股票投资的核心原则。")
            response = qwen_llm.invoke([message])
            
            print(f"✅ {model_name} 响应: {response.content}")
            
        except Exception as e:
            print(f"❌ {model_name} 测试失败: {e}")


def demonstrate_qwen_configuration():
    """
    演示如何配置使用 Qwen 模型
    """
    print("\n=== Qwen 配置演示 ===")
    
    config_example = """
# 在 config.yaml 中配置 Qwen:

api_keys:
  qwen: 'your-qwen-api-key-here'  # 替换为您的 Qwen API 密钥

llm_providers:
  qwen:
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    models:
      deep:
        - qwen-turbo
        - qwen-plus
        - qwen-max
        - qwen2.5-72b-instruct
        - qwen2.5-32b-instruct
      quick:
        - qwen-turbo
        - qwen2.5-7b-instruct
        - qwen2.5-14b-instruct

models:
  llm_provider: qwen
  deep_think_llm: qwen-max      # 用于复杂分析
  quick_think_llm: qwen-turbo   # 用于快速任务
    """
    
    print(config_example)


def main():
    """
    运行所有 Qwen 使用示例
    """
    print("🚀 Qwen 大模型使用示例\n")
    
    # 演示配置
    demonstrate_qwen_configuration()
    
    # 检查是否有 Qwen API 密钥
    config_manager = get_config_manager()
    qwen_api_key = config_manager.get_api_key('qwen')
    
    if not qwen_api_key or qwen_api_key == 'your-qwen-api-key-here':
        print("\n⚠️  请先在 config.yaml 中设置您的 Qwen API 密钥")
        print("设置完成后再运行此示例。")
        return
    
    # 运行测试
    tests = [
        test_qwen_basic_chat,
        test_qwen_financial_analysis,
        test_qwen_different_models
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ 测试 {test.__name__} 失败: {e}")
    
    print("\n🎉 Qwen 使用示例完成！")
    print("\n💡 提示:")
    print("- 使用 qwen-turbo 进行快速任务")
    print("- 使用 qwen-plus 进行平衡的性能和成本")
    print("- 使用 qwen-max 进行复杂分析")
    print("- 在 TradingAgents 中，您可以通过配置文件轻松切换模型")


if __name__ == '__main__':
    main()