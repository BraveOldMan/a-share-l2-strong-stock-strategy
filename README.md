# A-Share L2 Strong Stock Strategy

面向 A 股 Level-2 高频数据的强势股研究与 Walk-Forward 回测框架。

本仓库公开的是策略研究代码、因子工程、模型训练管线、回测状态机和历史说明文档。原始 Level-2 数据、训练缓存、模型二进制、日志和图表产物不随仓库发布。

> 免责声明：本项目仅用于量化研究、代码学习和回测框架参考，不构成任何投资建议。仓库不包含实盘交易入口，也不应被直接用于真实下单。

## 项目概览

该系统围绕 A 股 L2 Snapshot 十档盘口与 Tick 逐笔成交数据，提取订单簿失衡、流动性深度、成交侵略性、撤单异常、涨停磁吸、订单流毒性、SMC 聪明钱结构等高频微观特征。

模型层采用 Learning-to-Rank 思路，对每日股票截面进行相对强弱排序，并通过 LightGBM、XGBoost、CatBoost 等 Ranker 模型集成生成候选池。回测层使用 T+1 状态机、不可成交过滤、市场宽度风控、止损和移动止盈机制，尽量避免常见的未来函数和纸面成交问题。

核心流程：

```text
L2 Snapshot / Tick
  -> 单日微观结构因子
  -> 日内滚动特征与市场状态因子
  -> Alpha Mining 交叉因子挖掘
  -> LGB / XGB / CatBoost Ranker 集成排序
  -> 每日 Top 3 候选
  -> T+1 状态机回测
  -> 风控、止损、移动止盈、净值统计
```

## 仓库内容

| 路径 | 说明 |
| --- | --- |
| `batch_pipeline.py` | Walk-Forward 主流程，负责日期推进、缓存、训练、预测和状态机回测 |
| `v9_launcher.py` | Windows 推荐启动器，处理 `multiprocessing.freeze_support()` |
| `start_wf.py` | 简化版 Walk-Forward 启动脚本 |
| `run.py` | 早期单日因子计算、模型训练和 Top 3 输出入口 |
| `engine.py` | Snapshot 盘口因子与股票级/市场级聚合 |
| `tick_factors.py` | Tick 逐笔成交因子 |
| `custom_factors.py` | 高频盘口扩展因子 |
| `ml_pipeline.py` | 标签、滚动特征、模型训练、集成预测 |
| `factor_generator.py` | Alpha Mining 交叉因子挖掘 |
| `market_regime_gmm.py` | GMM 市场状态识别 |
| `universe_filter.py` | 股票池过滤和不可成交标的剔除 |
| `backtest_simulation.py` | T+1 状态机组合回测 |
| `strategy_documentation.md` | 系统架构说明和历史机制背景 |
| `CHANGELOG.md` | 版本演进记录 |
| `backtest_report_v20_zh.md` | V20 Walk-Forward 回测摘要 |

未发布到 GitHub 的内容：

- 原始 Level-2 Snapshot / Tick 数据
- `models/` 下的日缓存、模型文件和临时产物
- `catboost_info/` 训练日志
- 历史归档目录、压缩包、图表和本地日志
- 任何密钥、账户配置或实盘交易凭证

## 回测摘要

仓库内历史文档版本口径不完全一致：

- `strategy_documentation.md` 标注当前版本为 V18。
- `CHANGELOG.md` 标注最新运行版本为 V19。
- `backtest_report_v20_zh.md` 记录 V20 Walk-Forward 评估。

本 README 以 `backtest_report_v20_zh.md` 作为最新公开效果口径。

| 指标 | V20 结果 |
| --- | ---: |
| 测试周期 | 2024-11 至 2026-03-06 |
| 交易日数 | 322 |
| 总收益率 | +49.39% |
| 年化收益率 | 36.90% |
| 最大回撤 | -7.16% |
| Sharpe | 2.081 |
| 胜率 | 48.8% |
| Calmar | 5.156 |

这些结果依赖本地历史 L2 数据、缓存状态和当时的模型配置。由于数据没有随仓库发布，GitHub 上的代码不能单独复现上述结果。

## 环境要求

项目主要在 Windows + PowerShell 环境下开发。建议使用较新的 Python 3.x 环境。

仓库暂未提供 `requirements.txt`。源码中涉及的主要第三方库包括：

- `polars`
- `numpy`
- `scikit-learn`
- `lightgbm`
- `xgboost`
- `catboost`
- `joblib`
- `torch`
- `tqdm`
- `matplotlib`

如果需要复现，请先在隔离虚拟环境中安装依赖，并根据本机数据源调整路径。不要在生产环境或实盘环境中直接运行未审计代码。

## 数据准备

当前主流程在 `batch_pipeline.py` 中默认读取：

```text
E:/AGUDATA/l2_snapshot
E:/AGUDATA/l2_tick
```

期望的数据组织方式：

```text
l2_snapshot/
  date=YYYYMMDD/
    *.parquet

l2_tick/
  date=YYYYMMDD/
    *.parquet
```

如果你在其他目录保存数据，需要修改 `batch_pipeline.py` 中的：

```python
SNAPSHOT_BASE = "E:/AGUDATA/l2_snapshot"
TICK_BASE = "E:/AGUDATA/l2_tick"
```

注意：Level-2 数据通常存在版权和供应商许可限制。请只使用你有权访问和处理的数据。

## 快速开始

克隆仓库：

```powershell
git clone https://github.com/BraveOldMan/a-share-l2-strong-stock-strategy.git
Set-Location .\a-share-l2-strong-stock-strategy
```

创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

准备依赖后，推荐从 Walk-Forward 启动器开始：

```powershell
python .\v9_launcher.py
```

其他本地工具：

```powershell
python .\start_wf.py
python .\test_v21_generate_watchlist.py
python .\test_cache.py
python .\run.py 20240102
```

没有准备 L2 数据或 `models/daily_cache/` 的情况下，上述命令可能返回空结果或直接失败，这是预期行为。

## 设计重点

- 时间序列隔离：Walk-Forward 滚动训练和测试，避免把未来标签泄露到训练集。
- 排序学习：不直接预测绝对涨跌，而是预测同日截面中的相对强弱排序。
- 高频微观结构：重点利用盘口深度、逐笔成交、撤单、扫单、流动性和成交簇等信息。
- 市场状态过滤：通过市场宽度和状态识别，在系统性风险较高时降低或阻断开仓。
- 状态机回测：信号生成和资金执行解耦，模拟 T+1 pending order、持仓传导和跨日止盈止损。
- 产物隔离：数据、模型和日志默认不进入 Git 历史，避免公开大文件和受限数据。

## 研发规则

如果修改以下内容，应重新进行回测验证：

- 因子计算
- 标签生成
- 特征筛选
- 模型训练或预测逻辑
- 股票池过滤
- 仓位分配
- 止损止盈
- 回测撮合和资金状态机

验证时至少关注：

1. 是否存在未来函数。
2. 是否存在不可成交假设。
3. Sharpe 是否显著下降。
4. 最大回撤是否恶化。
5. 月度收益稳定性是否变差。
6. 修改是否只是过拟合某一段历史行情。

文档修改不要求完整回测，但应保持版本口径、数据边界和风险声明一致。

## 实盘红线

本仓库没有授权任何自动实盘交易行为。

禁止将本项目直接连接真实账户并自动下单。任何真实交易、账户接入、交易权限开启或实盘部署，都必须由使用者自行审计、理解风险并手动确认。

## License

当前仓库尚未声明开源许可证。公开可见不等于授予复制、修改、商用或再分发权利。

如需复用代码或策略思想，请先联系仓库所有者并确认许可边界。
