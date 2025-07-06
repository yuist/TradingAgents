# 多数据源系统使用指南

## 概述

多数据源系统是一个强大的数据获取和管理框架，支持从多个数据源获取股票数据，具有自动故障转移、数据质量验证和健康监控等功能。

## 主要特性

✅ **多数据源支持**：Yahoo Finance、Alpha Vantage、TuShare、新浪财经等
✅ **自动故障转移**：当一个数据源失败时，自动切换到备用数据源
✅ **数据质量验证**：全面的数据完整性、一致性和时效性检查
✅ **健康状态监控**：实时监控各数据源的健康状态和性能
✅ **灵活配置管理**：支持自定义数据源优先级和参数
✅ **无缝集成**：与现有智能体系统完全兼容

## 快速开始

### 1. 基本使用

```python
from tradingagents.dataflows.multi_source_manager import get_multi_source_manager

# 获取管理器实例
manager = get_multi_source_manager()

# 获取股票数据
data = manager.get_stock_data('AAPL', '2025-01-01', '2025-01-31')
print(f"获取到 {len(data)} 条数据")
```

### 2. 使用增强接口

```python
from tradingagents.dataflows.enhanced_interface import get_YFin_data_online_enhanced

# 获取CSV格式的股票数据
csv_data = get_YFin_data_online_enhanced('AAPL', '2025-01-01', '2025-01-31')
print(csv_data)
```

### 3. 便捷函数

```python
from tradingagents.dataflows.multi_source_manager import get_stock_data_with_fallback

# 带故障转移的数据获取
data = get_stock_data_with_fallback('AAPL', '2025-01-01', '2025-01-31')
```

## 高级功能

### 数据质量报告

```python
# 获取详细的数据质量报告
quality_report = manager.get_data_quality_report('AAPL', '2025-01-01', '2025-01-31')

print(f"质量分数: {quality_report['quality_score']:.1f}/100")
print(f"完整性: {quality_report['completeness']:.1f}/100")
print(f"一致性: {quality_report['consistency']:.1f}/100")
print(f"时效性: {quality_report['timeliness']:.1f}/100")

if quality_report['warnings']:
    print("警告信息:")
    for warning in quality_report['warnings']:
        print(f"  - {warning}")
```

### 健康状态监控

```python
# 获取所有数据源的健康状态
health_status = manager.get_health_status()

for market, providers in health_status.items():
    print(f"{market} 市场:")
    for provider in providers:
        print(f"  {provider['name']}: {provider['status']}")
        print(f"    成功率: {provider['success_rate']:.1f}%")
        print(f"    平均响应时间: {provider['avg_response_time']:.2f}s")

# 强制执行健康检查
manager.force_health_check()
```

### 实时数据获取

```python
# 获取实时股票数据
realtime_data = manager.get_realtime_data('AAPL')

print(f"当前价格: {realtime_data.get('current_price', 'N/A')}")
print(f"涨跌幅: {realtime_data.get('change_percent', 'N/A')}%")
print(f"成交量: {realtime_data.get('volume', 'N/A')}")
```

## 配置管理

### 查看默认配置

```python
from tradingagents.dataflows.data_source_config import get_default_config

config = get_default_config()
print(config)
```

### 自定义配置

```python
from tradingagents.dataflows.data_source_config import DataSourceConfig

# 加载配置
config_manager = DataSourceConfig('data_sources.json')

# 更新API密钥
config_manager.update_api_key('alpha_vantage', 'your_api_key_here')
config_manager.update_api_key('tushare', 'your_token_here')

# 设置数据源优先级
config_manager.set_priority('yahoo_finance', 1)  # 最高优先级
config_manager.set_priority('sina_finance', 2)

# 启用/禁用数据源
config_manager.enable_source('alpha_vantage')
config_manager.disable_source('polygon')

# 保存配置
config_manager.save_config()
```

## 支持的股票代码格式

### 国际市场
- 美股：`AAPL`, `TSLA`, `GOOGL`
- 其他国际股票：按照Yahoo Finance格式

### 中国市场
- 上海证券交易所：`600000.SH` 或 `600000`
- 深圳证券交易所：`000001.SZ` 或 `000001`
- 创业板：`300001.SZ` 或 `300001`
- 科创板：`688001.SH` 或 `688001`
- 北交所：`830001.BJ` 或 `830001`

## 错误处理

系统具有完善的错误处理机制：

```python
try:
    data = manager.get_stock_data('INVALID_SYMBOL', '2025-01-01', '2025-01-31')
except Exception as e:
    print(f"获取数据失败: {e}")
    
    # 查看健康状态以诊断问题
    health_status = manager.get_health_status()
    for market, providers in health_status.items():
        for provider in providers:
            if provider['status'] == 'failed':
                print(f"数据源 {provider['name']} 状态异常")
```

## 性能优化建议

1. **合理设置请求频率**：避免过于频繁的API调用
2. **使用缓存**：对于相同的数据请求，考虑使用缓存
3. **监控健康状态**：定期检查数据源健康状态
4. **配置优先级**：将最可靠的数据源设置为高优先级

## 故障排除

### 常见问题

1. **所有数据源都失败**
   - 检查网络连接
   - 验证API密钥是否正确
   - 查看健康状态报告

2. **数据质量分数低**
   - 检查数据源的数据完整性
   - 验证股票代码格式是否正确
   - 查看质量报告中的具体问题

3. **响应时间慢**
   - 检查网络状况
   - 考虑调整数据源优先级
   - 减少并发请求数量

### 调试模式

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 获取数据时会显示详细的调试信息
data = manager.get_stock_data('AAPL', '2025-01-01', '2025-01-31')
```

## 测试和验证

### 运行快速验证

```bash
python tradingagents/dataflows/quick_validation.py
```

### 运行完整测试

```bash
python tradingagents/dataflows/test_multi_source_system.py
```

### 运行演示

```bash
python tradingagents/dataflows/demo_multi_source.py
```

## 扩展和定制

### 添加新的数据源

1. 继承 `DataSourceInterface` 类
2. 实现必要的方法
3. 在配置文件中添加新数据源
4. 更新管理器初始化代码

### 自定义数据验证规则

可以扩展 `_validate_data_detailed` 方法来添加特定的验证逻辑。

## 支持和反馈

如果遇到问题或有改进建议，请：

1. 查看日志文件获取详细错误信息
2. 运行验证脚本检查系统状态
3. 检查配置文件是否正确
4. 确认网络连接和API密钥

---

**注意**：使用第三方API时请遵守相应的使用条款和频率限制。