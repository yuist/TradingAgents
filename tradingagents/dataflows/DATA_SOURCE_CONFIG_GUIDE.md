# 数据源配置指南

本指南将帮助您配置和管理TradingAgents系统的多数据源功能。

## 概述

TradingAgents支持多个金融数据源，包括免费和付费选项：

### 免费数据源
- **Yahoo Finance** - 全球股票数据，无需API密钥
- **新浪财经** - 中国股票数据，无需API密钥

### 付费数据源（需要API密钥）
- **Alpha Vantage** - 全球股票、外汇、加密货币数据
- **TuShare Pro** - 中国股票数据（专业版）
- **Polygon.io** - 美国股票数据

## 配置步骤

### 1. 基础配置

系统默认启用免费数据源（Yahoo Finance 和新浪财经），无需额外配置即可使用。

### 2. 配置付费数据源

如果您需要使用付费数据源，请按以下步骤配置：

#### 2.1 获取API密钥

**Alpha Vantage:**
1. 访问 https://www.alphavantage.co/support/#api-key
2. 注册账户并获取免费API密钥
3. 免费版限制：每分钟5次调用，每天500次调用

**TuShare Pro:**
1. 访问 https://tushare.pro/
2. 注册账户并获取Token
3. 需要积分才能使用高级功能

**Polygon.io:**
1. 访问 https://polygon.io/
2. 注册账户并获取API密钥
3. 免费版有调用限制

#### 2.2 配置API密钥

在项目根目录的 `config.yaml` 文件中添加您的API密钥：

```yaml
api_keys:
  # 现有配置...
  
  # 取消注释并填入您的API密钥
  # alpha_vantage_api_key: "YOUR_ALPHA_VANTAGE_API_KEY"
  # tushare_token: "YOUR_TUSHARE_TOKEN"
  # polygon_api_key: "YOUR_POLYGON_API_KEY"
```

#### 2.3 启用数据源

使用配置管理工具启用数据源：

```bash
# 查看当前状态
python tradingagents/dataflows/config_manager.py status

# 启用Alpha Vantage
python tradingagents/dataflows/config_manager.py enable alpha_vantage

# 启用TuShare
python tradingagents/dataflows/config_manager.py enable tushare

# 设置优先级（数字越小优先级越高）
python tradingagents/dataflows/config_manager.py priority alpha_vantage 1
```

## 数据源优先级

系统按优先级顺序尝试获取数据，当前默认优先级：

1. **优先级 1**: Alpha Vantage, TuShare（如果启用）
2. **优先级 2**: Yahoo Finance
3. **优先级 3**: 新浪财经

您可以根据需要调整优先级：

```bash
# 设置新浪财经为最高优先级（适合中国股票）
python tradingagents/dataflows/config_manager.py priority sina_finance 1

# 设置Yahoo Finance为次优先级
python tradingagents/dataflows/config_manager.py priority yahoo_finance 2
```

## 配置管理工具

使用内置的配置管理工具来管理数据源：

```bash
# 显示所有数据源状态
python tradingagents/dataflows/config_manager.py status

# 启用/禁用数据源
python tradingagents/dataflows/config_manager.py enable <source_name>
python tradingagents/dataflows/config_manager.py disable <source_name>

# 设置优先级
python tradingagents/dataflows/config_manager.py priority <source_name> <priority>

# 显示帮助
python tradingagents/dataflows/config_manager.py help
```

## 市场偏好设置

系统根据不同市场自动选择最适合的数据源：

- **国内市场**: TuShare → 新浪财经 → Yahoo Finance
- **国际市场**: Alpha Vantage → Polygon → Yahoo Finance
- **加密货币**: Alpha Vantage → Polygon

## 故障排除

### 常见问题

1. **API密钥无效**
   - 检查API密钥是否正确填写
   - 确认API密钥是否已激活
   - 检查是否超出调用限制

2. **数据获取失败**
   - 检查网络连接
   - 确认股票代码格式正确
   - 查看系统日志了解详细错误信息

3. **配置不生效**
   - 重启应用程序
   - 检查配置文件语法是否正确
   - 确认文件路径正确

### 调试模式

启用详细日志来诊断问题：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 最佳实践

1. **保持至少一个免费数据源启用**，确保系统始终可用
2. **根据主要交易市场设置优先级**，提高数据获取效率
3. **定期检查API密钥状态**，避免因过期导致的服务中断
4. **监控API调用限制**，避免超出免费额度
5. **备份配置文件**，便于快速恢复设置

## 配置文件结构

### 主配置文件 (config.yaml)
```yaml
api_keys:
  alpha_vantage_api_key: "your-key-here"
  tushare_token: "your-token-here"
  polygon_api_key: "your-key-here"
```

### 数据源配置文件 (data_sources.json)
```json
{
  "data_sources": {
    "alpha_vantage": {
      "enabled": false,
      "priority": 1,
      "api_key": "",
      "timeout": 30,
      "rate_limit": {
        "calls_per_minute": 5,
        "calls_per_day": 500
      }
    }
  }
}
```

## 支持

如果您在配置过程中遇到问题，请：

1. 查看系统日志文件
2. 运行配置验证工具
3. 检查API密钥是否有效
4. 确认网络连接正常

更多技术支持，请参考项目文档或提交Issue。