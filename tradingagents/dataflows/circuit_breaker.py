#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
熔断器模式实现
用于智能故障转移和避免重复尝试失败的数据源
"""

import time
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from ..utils.logging_manager import get_logger

logger = get_logger('dataflow', 'circuit_breaker')

class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常状态
    OPEN = "open"          # 熔断状态
    HALF_OPEN = "half_open" # 半开状态

@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 3      # 失败阈值
    timeout_duration: int = 1800    # 熔断超时时间（秒）30分钟
    success_threshold: int = 2      # 半开状态成功阈值
    
class CircuitBreaker:
    """熔断器实现"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        
    def can_execute(self) -> bool:
        """检查是否可以执行请求"""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # 检查是否可以转为半开状态
            if current_time - self.last_failure_time >= self.config.timeout_duration:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"熔断器 {self.name} 转为半开状态")
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self):
        """记录成功"""
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"熔断器 {self.name} 恢复正常状态")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self):
        """记录失败"""
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"熔断器 {self.name} 触发熔断，失败次数: {self.failure_count}")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning(f"熔断器 {self.name} 重新熔断")
    
    def get_status(self) -> Dict[str, Any]:
        """获取熔断器状态"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time,
            'can_execute': self.can_execute()
        }
    
    def reset(self):
        """重置熔断器"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"熔断器 {self.name} 已重置")

class CircuitBreakerManager:
    """熔断器管理器"""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = CircuitBreakerConfig()
    
    def get_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """获取或创建熔断器"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config or self.default_config)
        return self.breakers[name]
    
    def can_execute(self, name: str) -> bool:
        """检查指定熔断器是否可以执行"""
        breaker = self.get_breaker(name)
        return breaker.can_execute()
    
    def record_success(self, name: str):
        """记录成功"""
        breaker = self.get_breaker(name)
        breaker.record_success()
    
    def record_failure(self, name: str):
        """记录失败"""
        breaker = self.get_breaker(name)
        breaker.record_failure()
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有熔断器状态"""
        return {name: breaker.get_status() for name, breaker in self.breakers.items()}
    
    def reset_breaker(self, name: str):
        """重置指定熔断器"""
        if name in self.breakers:
            self.breakers[name].reset()
    
    def reset_all(self):
        """重置所有熔断器"""
        for breaker in self.breakers.values():
            breaker.reset()

# 全局熔断器管理器实例
_global_circuit_breaker_manager: Optional[CircuitBreakerManager] = None

def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """获取全局熔断器管理器"""
    global _global_circuit_breaker_manager
    if _global_circuit_breaker_manager is None:
        _global_circuit_breaker_manager = CircuitBreakerManager()
    return _global_circuit_breaker_manager