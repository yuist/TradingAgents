#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据缓存机制
用于缓存股票数据，减少重复请求
"""

import time
import hashlib
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ..utils.logging_manager import get_logger

logger = get_logger('dataflow', 'data_cache')

@dataclass
class CacheEntry:
    """缓存条目"""
    data: pd.DataFrame
    timestamp: float
    source: str
    ttl: int  # 生存时间（秒）
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() - self.timestamp > self.ttl
    
    def get_age(self) -> int:
        """获取缓存年龄（秒）"""
        return int(time.time() - self.timestamp)

@dataclass
class RealtimeCacheEntry:
    """实时数据缓存条目"""
    data: Dict[str, Any]
    timestamp: float
    source: str
    ttl: int  # 生存时间（秒）
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() - self.timestamp > self.ttl
    
    def get_age(self) -> int:
        """获取缓存年龄（秒）"""
        return int(time.time() - self.timestamp)

class DataCache:
    """数据缓存管理器"""
    
    def __init__(self, default_ttl: int = 900):  # 默认15分钟
        self.cache: Dict[str, CacheEntry] = {}
        self.realtime_cache: Dict[str, RealtimeCacheEntry] = {}
        self.default_ttl = default_ttl
        self.realtime_ttl = 60  # 实时数据默认1分钟TTL
        self.hit_count = 0
        self.miss_count = 0
        self.realtime_hit_count = 0
        self.realtime_miss_count = 0
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5分钟清理一次
    
    def _generate_key(self, symbol: str, start_date: str, end_date: str) -> str:
        """生成缓存键"""
        key_string = f"{symbol}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cleanup_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        # 清理历史数据缓存
        expired_keys = []
        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        # 清理实时数据缓存
        expired_realtime_keys = []
        for key, entry in self.realtime_cache.items():
            if entry.is_expired():
                expired_realtime_keys.append(key)
        
        for key in expired_realtime_keys:
            del self.realtime_cache[key]
        
        total_expired = len(expired_keys) + len(expired_realtime_keys)
        if total_expired > 0:
            logger.debug(f"清理了 {total_expired} 个过期缓存条目 (历史: {len(expired_keys)}, 实时: {len(expired_realtime_keys)})")
        
        self.last_cleanup = current_time
    
    def get(self, symbol: str, start_date: str, end_date: str) -> Optional[Tuple[pd.DataFrame, str, int]]:
        """获取缓存数据"""
        self._cleanup_expired()
        
        key = self._generate_key(symbol, start_date, end_date)
        
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                self.hit_count += 1
                logger.debug(f"缓存命中: {symbol} ({start_date} 到 {end_date}), 来源: {entry.source}, 年龄: {entry.get_age()}秒")
                return entry.data.copy(), entry.source, entry.get_age()
            else:
                # 过期缓存
                del self.cache[key]
        
        self.miss_count += 1
        logger.debug(f"缓存未命中: {symbol} ({start_date} 到 {end_date})")
        return None
    
    def put(self, symbol: str, start_date: str, end_date: str, data: pd.DataFrame, 
            source: str, ttl: Optional[int] = None):
        """存储数据到缓存"""
        if data.empty:
            return
        
        key = self._generate_key(symbol, start_date, end_date)
        ttl = ttl or self.default_ttl
        
        entry = CacheEntry(
            data=data.copy(),
            timestamp=time.time(),
            source=source,
            ttl=ttl
        )
        
        self.cache[key] = entry
        logger.debug(f"缓存存储: {symbol} ({start_date} 到 {end_date}), 来源: {source}, TTL: {ttl}秒")
    
    def invalidate(self, symbol: str = None, start_date: str = None, end_date: str = None):
        """使缓存失效"""
        if symbol and start_date and end_date:
            # 删除特定缓存
            key = self._generate_key(symbol, start_date, end_date)
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"缓存失效: {symbol} ({start_date} 到 {end_date})")
        else:
            # 清空所有缓存
            count = len(self.cache)
            self.cache.clear()
            logger.info(f"清空所有缓存，共 {count} 个条目")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        # 历史数据统计
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        # 实时数据统计
        total_realtime_requests = self.realtime_hit_count + self.realtime_miss_count
        realtime_hit_rate = (self.realtime_hit_count / total_realtime_requests * 100) if total_realtime_requests > 0 else 0
        
        # 统计历史数据缓存条目信息
        active_entries = 0
        expired_entries = 0
        sources = {}
        
        for entry in self.cache.values():
            if entry.is_expired():
                expired_entries += 1
            else:
                active_entries += 1
                sources[entry.source] = sources.get(entry.source, 0) + 1
        
        # 统计实时数据缓存条目信息
        active_realtime_entries = 0
        expired_realtime_entries = 0
        realtime_sources = {}
        
        for entry in self.realtime_cache.values():
            if entry.is_expired():
                expired_realtime_entries += 1
            else:
                active_realtime_entries += 1
                realtime_sources[entry.source] = realtime_sources.get(entry.source, 0) + 1
        
        return {
            'historical': {
                'total_entries': len(self.cache),
                'active_entries': active_entries,
                'expired_entries': expired_entries,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': round(hit_rate, 2),
                'sources': sources,
                'default_ttl': self.default_ttl
            },
            'realtime': {
                'total_entries': len(self.realtime_cache),
                'active_entries': active_realtime_entries,
                'expired_entries': expired_realtime_entries,
                'hit_count': self.realtime_hit_count,
                'miss_count': self.realtime_miss_count,
                'hit_rate': round(realtime_hit_rate, 2),
                'sources': realtime_sources,
                'default_ttl': self.realtime_ttl
            }
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取详细的缓存信息"""
        cache_info = []
        
        for key, entry in self.cache.items():
            cache_info.append({
                'key': key,
                'source': entry.source,
                'timestamp': entry.timestamp,
                'age_seconds': entry.get_age(),
                'ttl': entry.ttl,
                'expired': entry.is_expired(),
                'data_points': len(entry.data)
            })
        
        return {
            'entries': cache_info,
            'stats': self.get_stats()
        }
    
    def get_realtime(self, symbol: str) -> Optional[Tuple[Dict[str, Any], str, int]]:
        """获取实时数据缓存"""
        self._cleanup_expired()
        
        if symbol in self.realtime_cache:
            entry = self.realtime_cache[symbol]
            if not entry.is_expired():
                self.realtime_hit_count += 1
                logger.debug(f"实时缓存命中: {symbol}, 来源: {entry.source}, 年龄: {entry.get_age()}秒")
                return entry.data.copy(), entry.source, entry.get_age()
            else:
                # 过期缓存
                del self.realtime_cache[symbol]
        
        self.realtime_miss_count += 1
        logger.debug(f"实时缓存未命中: {symbol}")
        return None
    
    def put_realtime(self, symbol: str, data: Dict[str, Any], source: str, ttl: Optional[int] = None):
        """存储实时数据到缓存"""
        if not data:
            return
        
        ttl = ttl or self.realtime_ttl
        
        entry = RealtimeCacheEntry(
            data=data.copy(),
            timestamp=time.time(),
            source=source,
            ttl=ttl
        )
        
        self.realtime_cache[symbol] = entry
        logger.debug(f"实时缓存存储: {symbol}, 来源: {source}, TTL: {ttl}秒")
    
    def invalidate_realtime(self, symbol: str = None):
        """使实时缓存失效"""
        if symbol:
            # 删除特定缓存
            if symbol in self.realtime_cache:
                del self.realtime_cache[symbol]
                logger.debug(f"实时缓存失效: {symbol}")
        else:
            # 清空所有实时缓存
            count = len(self.realtime_cache)
            self.realtime_cache.clear()
            logger.info(f"清空所有实时缓存，共 {count} 个条目")
    
    def set_default_ttl(self, ttl: int):
        """设置默认TTL"""
        self.default_ttl = ttl
        logger.info(f"默认TTL设置为 {ttl} 秒")
    
    def set_realtime_ttl(self, ttl: int):
        """设置实时数据默认生存时间"""
        self.realtime_ttl = ttl
    
    def clear(self):
        """清空所有缓存"""
        self.cache.clear()
        self.realtime_cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.realtime_hit_count = 0
        self.realtime_miss_count = 0
        logger.info("缓存已清空")

# 全局缓存实例
_global_data_cache: Optional[DataCache] = None

def get_data_cache() -> DataCache:
    """获取全局数据缓存实例"""
    global _global_data_cache
    if _global_data_cache is None:
        _global_data_cache = DataCache()
    return _global_data_cache

def clear_cache():
    """清空全局缓存"""
    cache = get_data_cache()
    cache.invalidate()

def get_cache_stats() -> Dict[str, Any]:
    """获取缓存统计信息"""
    cache = get_data_cache()
    return cache.get_stats()