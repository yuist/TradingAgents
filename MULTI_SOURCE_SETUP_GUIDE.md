# 多数据源系统安装和部署指南

## 概述

本指南将帮助您快速设置和部署多数据源股票数据获取系统，实现从单一Yahoo Finance依赖到多数据源备用的升级。

## 系统架构

```
应用层
├── enhanced_interface.py (统一接口)
├── multi_source_manager.py (核心管理器)
└── data_source_config.py (配置管理)

数据源层
├── Yahoo Finance (免费，国际股票)
├── Alpha Vantage (免费额度，高质量)
├── TuShare Pro (国内股票，需注册)
├── 新浪财经 (免费，国内股票)
└── Polygon (付费，高频数据)
```

## 快速开始

### 1. 文件部署

将以下文件复制到您的项目中：

```
tradingagents/dataflows/
├── multi_source_manager.py      # 核心多数据源管理器
├── enhanced_interface.py        # 增强的统一接口
├── data_source_config.py        # 配置管理
└── data_sources.json           # 配置文件（可选）
```

### 2. 依赖安装

```bash
# 基础依赖
pip install pandas requests

# 可选数据源依赖
pip install tushare              # TuShare Pro
pip install alpha-vantage        # Alpha Vantage
pip install polygon-api-client   # Polygon
```

### 3. 基础配置

#### 方法一：使用配置文件

```python
# 生成示例配置文件
from tradingagents.dataflows.data_source_config import DataSourceConfig

config_manager = DataSourceConfig()
config_manager.create_sample_config()
```

编辑生成的 `data_sources_sample.json`，填入您的API密钥：

```json
{
  "data_sources": {
    "alpha_vantage": {
      "enabled": true,
      "api_key": "YOUR_ALPHA_VANTAGE_API_KEY"
    },
    "tushare": {
      "enabled": true,
      "token": "YOUR_TUSHARE_TOKEN"
    }
  }
}
```

#### 方法二：使用环境变量

```bash
# Windows
set ALPHA_VANTAGE_API_KEY=your_api_key_here
set TUSHARE_TOKEN=your_token_here

# Linux/Mac
export ALPHA_VANTAGE_API_KEY=your_api_key_here
export TUSHARE_TOKEN=your_token_here
```

### 4. 快速测试

```python
# 运行测试脚本
python test_multi_source.py
```

## 详细配置

### 数据源配置

#### 1. Yahoo Finance（默认启用）
- **优点**：免费，覆盖全球股票
- **缺点**：频率限制，稳定性一般
- **配置**：无需API密钥

#### 2. Alpha Vantage（推荐）
- **优点**：数据质量高，稳定性好
- **缺点**：免费额度有限（每分钟5次，每天500次）
- **获取API密钥**：[https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)

```python
# 配置Alpha Vantage
from tradingagents.dataflows.data_source_config import setup_api_keys

setup_api_keys(alpha_vantage="YOUR_API_KEY")
```

#### 3. TuShare Pro（国内股票推荐）
- **优点**：国内股票数据全面，更新及时
- **缺点**：需要注册和积分
- **注册地址**：[https://tushare.pro/register](https://tushare.pro/register)

```python
# 配置TuShare
setup_api_keys(tushare="YOUR_TUSHARE_TOKEN")
```

#### 4. 新浪财经（默认启用）
- **优点**：免费，国内股票实时数据
- **缺点**：非官方API，稳定性不保证
- **配置**：无需API密钥

#### 5. Polygon（高级用户）
- **优点**：高频数据，企业级稳定性
- **缺点**：付费服务
- **获取API密钥**：[https://polygon.io/](https://polygon.io/)

### 优先级配置

数据源按优先级排序（数字越小优先级越高）：

```json
{
  "data_sources": {
    "alpha_vantage": {"priority": 1},
    "tushare": {"priority": 1},
    "yahoo_finance": {"priority": 2},
    "sina_finance": {"priority": 3}
  }
}
```

## 使用方法

### 1. 替换现有接口

将原有的数据获取代码：

```python
# 原有代码
from tradingagents.dataflows.interface import get_YFin_data_online

data = get_YFin_data_online("AAPL", "2024-01-01", "2024-01-07")
```

替换为：

```python
# 新代码（兼容原接口）
from tradingagents.dataflows.enhanced_interface import get_YFin_data_online_enhanced

data = get_YFin_data_online_enhanced("AAPL", "2024-01-01", "2024-01-07")
```

### 2. 使用新的DataFrame接口

```python
from tradingagents.dataflows.enhanced_interface import get_stock_dataframe

# 直接获取DataFrame
df = get_stock_dataframe("AAPL", "2024-01-01", "2024-01-07")
print(df.head())
```

### 3. 健康监控

```python
from tradingagents.dataflows.enhanced_interface import check_data_sources_health

# 检查数据源状态
health = check_data_sources_health()
print(health)
```

### 4. 性能对比

```python
from tradingagents.dataflows.enhanced_interface import benchmark_data_sources

# 对比不同数据源性能
benchmark = benchmark_data_sources("AAPL")
for source, result in benchmark.items():
    print(f"{source}: {result}")
```

### 5. 强制使用特定数据源

```python
from tradingagents.dataflows.enhanced_interface import get_enhanced_interface

interface = get_enhanced_interface()

# 强制使用TuShare获取数据
data = interface.switch_to_source("tushare", "000001", "2024-01-01", "2024-01-07")
```

## 故障排除

### 常见问题

#### 1. 导入错误

```
ImportError: 多数据源管理器未正确安装
```

**解决方案**：
- 确保文件路径正确
- 检查Python路径设置
- 重新复制文件

#### 2. API密钥错误

```
配置警告: ['alpha_vantage 已启用但未配置API密钥']
```

**解决方案**：
- 检查API密钥是否正确
- 确认配置文件格式
- 使用环境变量设置

#### 3. 网络连接问题

```
所有数据源均无法获取 AAPL 的数据
```

**解决方案**：
- 检查网络连接
- 确认防火墙设置
- 尝试使用代理

#### 4. 频率限制

```
YFRateLimitError: Rate limit exceeded
```

**解决方案**：
- 系统会自动切换到备用数据源
- 调整请求频率
- 升级到付费API

### 调试模式

启用详细日志：

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('tradingagents.dataflows')
logger.setLevel(logging.DEBUG)
```

### 性能优化

#### 1. 缓存配置

```json
{
  "fallback_strategy": {
    "enable_cache": true,
    "cache_duration_hours": 24
  }
}
```

#### 2. 并发控制

```json
{
  "rate_limit": {
    "calls_per_minute": 60,
    "concurrent_requests": 3
  }
}
```

## 监控和维护

### 1. 健康检查

建议定期执行健康检查：

```python
# 每小时执行一次
import schedule
import time

def health_check():
    from tradingagents.dataflows.enhanced_interface import get_enhanced_interface
    interface = get_enhanced_interface()
    interface.force_health_check()
    print(f"健康检查完成: {time.strftime('%Y-%m-%d %H:%M:%S')}")

schedule.every().hour.do(health_check)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 2. 日志监控

监控关键日志：

```python
import logging

# 设置日志文件
logging.basicConfig(
    filename='data_sources.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 3. 性能指标

定期收集性能指标：

```python
def collect_metrics():
    from tradingagents.dataflows.enhanced_interface import benchmark_data_sources
    
    results = benchmark_data_sources()
    
    # 保存到文件或数据库
    with open('performance_metrics.json', 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'results': results
        }, f)
```

## 升级和迁移

### 从单一Yahoo Finance升级

1. **备份现有代码**
2. **部署新文件**
3. **更新导入语句**
4. **配置数据源**
5. **测试验证**
6. **逐步切换**

### 迁移检查清单

- [ ] 备份原有代码
- [ ] 部署多数据源文件
- [ ] 配置API密钥
- [ ] 运行测试脚本
- [ ] 更新导入语句
- [ ] 验证数据一致性
- [ ] 监控系统运行
- [ ] 文档更新

## 最佳实践

### 1. 数据源选择

- **国际股票**：Alpha Vantage > Yahoo Finance
- **国内股票**：TuShare > 新浪财经 > Yahoo Finance
- **实时数据**：新浪财经 > TuShare
- **历史数据**：Alpha Vantage > TuShare > Yahoo Finance

### 2. 错误处理

```python
try:
    data = get_stock_dataframe("AAPL", "2024-01-01", "2024-01-07")
except Exception as e:
    logger.error(f"数据获取失败: {e}")
    # 实施备用策略
```

### 3. 配置管理

- 使用环境变量存储敏感信息
- 定期备份配置文件
- 版本控制配置变更

### 4. 监控告警

- 设置数据源失败告警
- 监控API配额使用
- 跟踪响应时间变化

## 技术支持

### 联系方式

- **问题反馈**：通过GitHub Issues
- **功能建议**：提交Pull Request
- **技术讨论**：项目讨论区

### 常用资源

- [Alpha Vantage文档](https://www.alphavantage.co/documentation/)
- [TuShare文档](https://tushare.pro/document/2)
- [Pandas文档](https://pandas.pydata.org/docs/)
- [Python日志文档](https://docs.python.org/3/library/logging.html)

---

**注意**：本系统设计为渐进式升级，您可以先部署基础功能，然后逐步添加更多数据源和高级特性。