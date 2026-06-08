# A股 L2 强势股策略

基于 A 股 Level-2 Snapshot 十档盘口与 Tick 逐笔成交数据的强势股研究、因子挖掘与 Walk-Forward 回测系统。

本项目目标不是直接实盘交易，而是围绕高频微观结构因子、排序学习模型和严格状态机回测，验证次日强势股候选的可行性与风险边界。

## 当前版本口径

仓库内历史说明文件存在版本口径不一致：

- `strategy_documentation.md` 标注当前版本为 V18。
- `CHANGELOG.md` 标注最新运行版本为 V19。
- `backtest_report_v20_zh.md` 给出了最新 V20 Walk-Forward 回测评估。

因此，本 README 以 **V20 回测报告** 作为当前效果口径，以 `strategy_documentation.md` 和 `CHANGELOG.md` 作为架构与演进背景资料。

V20 样本外 Walk-Forward 回测摘要：

| 指标 | 数值 |
| --- | ---: |
| 测试周期 | 2024-11 至 2026-03-06 |
| 交易日数 | 322 |
| 总收益率 | +49.39% |
| 年化收益率 | 36.90% |
| 最大回撤 | -7.16% |
| Sharpe | 2.081 |
| 胜率 | 48.8% |
| Calmar | 5.156 |

## 核心思路

系统从 L2 高频数据中提取盘口深度、订单流、成交侵略性、流动性毒性、涨停磁吸、SMC 聪明钱结构等微观特征，再通过 Learning-to-Rank 排序学习模型对每日股票截面进行相对强弱排序。

核心流程：

```text
L2 Snapshot / Tick
  -> 单日因子聚合与缓存
  -> 滚动特征与市场状态因子
  -> Alpha Mining 交叉因子挖掘
  -> LightGBM / XGBoost / CatBoost Ranker 集成
  -> 每日 Top 3 候选
  -> T+1 状态机模拟成交
  -> 止损、移动止盈、市场宽度空仓保护
```

主要风控原则：

- 避免未来函数：训练与测试按时间滚动推进，Walk-Forward 中使用 Purge / Embargo 思路隔离近期标签泄漏。
- 拒绝纸面成交：`universe_filter.py` 会过滤 ST、异常价格、一字涨停等不可实际买入标的。
- 控制系统性风险：Market Breadth 市场宽度过滤器在弱势环境下阻断新开仓。
- 资金状态独立：`StatefulPortfolioManager` 使用 T+1 pending order 队列，避免把预测日收益当作当日可交易收益。
- 让利润奔跑：持仓通过最高净值回撤阈值执行移动止盈，同时保留刚性止损。

## 目录结构

| 路径 | 作用 |
| --- | --- |
| `batch_pipeline.py` | Walk-Forward 主回测管线，负责日期推进、缓存、训练、预测和状态机回测 |
| `v9_launcher.py` | Windows 下推荐的 Walk-Forward 启动器，包含 `multiprocessing.freeze_support()` |
| `start_wf.py` | 简单 Walk-Forward 启动脚本 |
| `run.py` | 早期单日因子、训练、Top 3 输出和模拟回测入口 |
| `engine.py` | Snapshot 盘口因子、股票级和市场级聚合 |
| `tick_factors.py` | Tick 逐笔成交和订单流因子 |
| `custom_factors.py` | 高频盘口扩展因子 |
| `ml_pipeline.py` | 标签生成、滚动特征、模型训练和集成预测 |
| `factor_generator.py` | Alpha Mining 交叉因子挖掘 |
| `market_regime_gmm.py` | 市场状态聚类 |
| `universe_filter.py` | 股票池过滤和不可交易标的剔除 |
| `backtest_simulation.py` | 状态机组合管理与回测成交模拟 |
| `models/` | 模型、缓存、因子配置和回测输出 |
| `models/daily_cache/` | 单日截面特征缓存 |
| `backtest_report_v20_zh.md` | V20 Walk-Forward 回测评估报告 |
| `strategy_documentation.md` | 系统架构和历史机制说明 |
| `CHANGELOG.md` | 版本演进记录 |

## 数据路径

主回测管线当前在 `batch_pipeline.py` 中使用以下数据源路径：

```text
E:/AGUDATA/l2_snapshot
E:/AGUDATA/l2_tick
```

Snapshot 数据按交易日目录组织：

```text
E:/AGUDATA/l2_snapshot/date=YYYYMMDD/*.parquet
```

Tick 数据由 `tick_factors.py` 读取并按交易日融合。

如果原始 Snapshot 路径不可用，`get_all_dates()` 会尝试从 `models/daily_cache/*.parquet` 推导可用交易日，但这只能复用已有缓存，不能替代完整原始数据回放。

## 快速运行

在 Windows PowerShell 中进入项目目录：

```powershell
Set-Location "D:\量化策略\FactorGeneration\A股L2强势股策略"
```

推荐启动 Walk-Forward 回测：

```powershell
python .\v9_launcher.py
```

也可以使用默认 Walk-Forward 入口：

```powershell
python .\start_wf.py
```

运行较短窗口的观察列表生成测试：

```powershell
python .\test_v21_generate_watchlist.py
```

检查日缓存是否能读取：

```powershell
python .\test_cache.py
```

早期单日演示入口：

```powershell
python .\run.py 20240102
```

注意：仓库未提供 `requirements.txt`。不要为了运行随意新增依赖；如环境缺包，应先确认当前项目实际依赖和已有 Python 环境。

## 主要输出

常见输出位于 `models/`：

| 输出 | 说明 |
| --- | --- |
| `models/daily_cache/*.parquet` | 单日特征截面缓存 |
| `models/oos_equity_curve.csv` | OOS 回测净值曲线 |
| `models/wf_crash_error.log` | Walk-Forward 异常日志 |
| `models/discovered_factors.json` | Alpha Mining 发现因子 |
| `models/active_features.json` | 当前启用特征 |
| `models/dead_factors.json` | 被剔除的失效因子 |
| `models/feature_importance.json` | 特征重要性摘要 |

仓库顶层也保留了若干历史评估产物，例如 `wf_equity_curve.csv`、`wf_pnl.log`、`wf_benchmark_plot.png` 和多个版本回测报告。

## 研发与验证规则

修改策略逻辑、特征、标签、过滤器、仓位或回测撮合规则后，必须重新跑回测验证。

最低验证要求：

1. 能复现触发路径或目标问题。
2. Walk-Forward 或最小窗口回测可以正常跑完。
3. 对比修改前后核心 KPI，至少关注 Sharpe、最大回撤、总收益、胜率和月度稳定性。
4. 如果 Sharpe 明显下降，默认回滚该策略逻辑修改，除非有明确的风险收益取舍说明。
5. 主动检查未来函数、过拟合和不可成交假设。

文档或非策略逻辑修改不需要完整回测，但需要确认 README、报告和版本口径没有互相矛盾。

## 实盘红线

本仓库仅用于研究、回测和模拟验证。禁止通过自动化方式启动任何实盘交易命令。

任何实盘部署、真实下单、账户接入或交易权限开启，都必须由人工单独确认并手动执行。

## 参考文档

- `strategy_documentation.md`：完整系统说明和 V18 架构背景。
- `CHANGELOG.md`：V1 到 V19 的核心改动。
- `backtest_report_v20_zh.md`：V20 Walk-Forward 回测表现。
- `backtest_report_v15_zh.md`、`backtest_report_v15_lgbm_zh.md`：历史版本回测资料。
