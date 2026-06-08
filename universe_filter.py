"""
选股宇宙过滤器 V8
==================
在模型预测前剔除低质量标的，确保选股池的纯净度。
[V8 升级]:
- 新增科创板 (688xxx) / 北交所 (8xxxxx) 代码排除
- 新增次新股过滤 (上市不足 60 个交易日)
"""
import polars as pl


def filter_universe(df: pl.DataFrame) -> pl.DataFrame:
    """
    过滤低质量标的，确保选股池纯净。

    过滤规则:
    1. 剔除 ST/*ST 股 (万得代码中含 'ST' 前缀的)
    2. 剔除停牌股 (成交价为 0 或 last_price 为 0)
    3. 剔除零成交量标的 (涨跌幅 ≈ 0 且无成交信号)
    4. [V8] 排除科创板 688xxx.SH (20% 涨跌幅不适用 10% 模型假设)
    5. [V8] 排除北交所 8xxxxx.BJ (30% 涨跌幅不适用)
    6. [V8] 排除次新股 (该代码在历史数据中出现天数 < 60)
    7. 涨停一字板过滤 (一字涨停无法买入)
    """
    before = df.height

    # 1. 基础代码格式过滤 + ST 近似排除
    if "万得代码" in df.columns:
        df = df.filter(
            pl.col("万得代码").str.contains(r"^\d{6}\.[A-Z]{2}$")
        )
        
        # [V17.0] 终极合卷: 恢复大盘全集(主板+创业板) 跷跷板自由轮动环境
        # 60: 沪市主板; 00: 深市主板/中小板; 30: 创业板
        df = df.filter(
            pl.col("万得代码").str.starts_with("60") |
            pl.col("万得代码").str.starts_with("00") |
            pl.col("万得代码").str.starts_with("30")
        )

    # 2. 停牌股 / 零成交过滤
    if "last_price" in df.columns:
        df = df.filter(pl.col("last_price") > 0)

    if "ofi_sum" in df.columns:
        df = df.filter(pl.col("ofi_sum").abs() > 1e-9)

    # [V15] 流动性下限过滤: 必须满足基础交易活跃度，淘汰微盘死水股
    if "当日累计成交量" in df.columns:
        # 假设累计成交量单位为股，筛掉全天成交少于 200万股 的极度冷门股
        df = df.filter(pl.col("当日累计成交量") >= 2_000_000)

    # [V8] 6. 次新股过滤: 如果该代码在数据中出现的交易日数 < 60，剔除
    # [BugFix] 仅当数据包含多个交易日时才有意义；截面数据（单日）跳过此过滤
    if "trade_date" in df.columns and "万得代码" in df.columns:
        n_dates = df.select("trade_date").n_unique()
        if n_dates > 1:
            stock_day_counts = df.group_by("万得代码").agg(
                pl.col("trade_date").n_unique().alias("n_trade_days")
            )
            mature_stocks = stock_day_counts.filter(pl.col("n_trade_days") >= 60).select("万得代码")
            if mature_stocks.height > 0:
                df = df.join(mature_stocks, on="万得代码", how="semi")

    # 7. 涨停一字板过滤 (一字涨停无法买入)
    if "touched_limit_up" in df.columns and "tick_count" in df.columns:
        df = df.filter(
            ~((pl.col("touched_limit_up") == 1) & (pl.col("tick_count") < 5))
        )

    after = df.height
    filtered = before - after
    if filtered > 0:
        print(f"    [Universe] 过滤 {filtered} 只低质量标的 ({before} -> {after})")

    return df
