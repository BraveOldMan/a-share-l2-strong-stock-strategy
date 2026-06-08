import polars as pl
import numpy as np

from custom_factors import (
    factor_ofi_multi_level_ashare,
    factor_volatility_ashare,
    factor_simple_obi_ashare,
    factor_weighted_price_diff_ashare,
    factor_depth_variance,
    factor_book_slope,
    factor_depth_weighted_impact_ashare,
    factor_obi_momentum_ashare,
    factor_liquidity_void_probe
)

def format_datetime(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("自然日").cast(pl.Int64).cast(pl.Utf8).alias("date_str"),
        pl.col("时间").cast(pl.Int64).cast(pl.Utf8).str.zfill(9).alias("time_str")
    )
    df = df.with_columns(
        (pl.col("date_str") + " " + pl.col("time_str")).alias("datetime_str")
    )
    df = df.with_columns(
        pl.col("datetime_str").str.strptime(pl.Datetime, format="%Y%m%d %H%M%S%3f", strict=False).alias("datetime")
    ).drop(["date_str", "time_str", "datetime_str"])
    df = df.filter(pl.col("datetime").is_not_null())
    return df

def assign_session_id(df: pl.DataFrame) -> pl.DataFrame:
    """
    V3: 分时段标记
    0 = 集合竞价 (09:15-09:25)
    1 = 早盘爆发期 (09:30-10:00) 
    2 = 上午盘中 (10:00-11:30)
    3 = 午后反包 (13:00-14:00)
    4 = 尾盘定盘 (14:00-15:00)
    """
    df = df.with_columns(
        pl.col("datetime").dt.hour().alias("_hour"),
        pl.col("datetime").dt.minute().alias("_minute")
    )
    
    df = df.with_columns(
        pl.when((pl.col("_hour") == 9) & (pl.col("_minute") < 25))
        .then(pl.lit(0))
        .when((pl.col("_hour") == 9) & (pl.col("_minute") >= 30))
        .then(pl.lit(1))
        .when((pl.col("_hour") >= 10) & (pl.col("_hour") < 12))
        .then(pl.lit(2))
        .when((pl.col("_hour") == 13) & (pl.col("_minute") < 60))
        .then(pl.lit(3))
        .when(pl.col("_hour") >= 14)
        .then(pl.lit(4))
        .otherwise(pl.lit(2))
        .alias("session_id")
    ).drop(["_hour", "_minute"])
    
    return df

def compute_stock_level_factors(df_combined: pl.DataFrame, time_step_seconds: int = 60) -> pl.DataFrame:
    """第一阶段：针对个股计算因子并进行时间聚合"""
    if df_combined.height == 0:
        return pl.DataFrame()
        
    rename_map = {}
    if "前收盘" in df_combined.columns:
        rename_map["前收盘"] = "pre_close"
    if "成交价" in df_combined.columns:
        rename_map["成交价"] = "last_price"
    if rename_map:
        df_combined = df_combined.rename(rename_map)
        
    # 1. 数据转换
    df_combined = format_datetime(df_combined)
    
    price_cols = [f'申买价{i}' for i in range(1, 11)] + [f'申卖价{i}' for i in range(1, 11)] + ['pre_close', 'last_price']
    df_combined = df_combined.with_columns(
        [(pl.col(c) / 10000.0).cast(pl.Float32) for c in price_cols if c in df_combined.columns]
    )
    
    # 时段标记
    df_combined = assign_session_id(df_combined)
    df_combined = df_combined.sort(["datetime", "万得代码"])
    
    # 准备矩阵
    bp_cols = [f'申买价{i}' for i in range(1, 11)]
    ba_cols = [f'申买量{i}' for i in range(1, 11)]
    ap_cols = [f'申卖价{i}' for i in range(1, 11)]
    aa_cols = [f'申卖量{i}' for i in range(1, 11)]
    
    # 提前计算十档委托量总和供 OIR_10 使用
    valid_ba = [c for c in ba_cols if c in df_combined.columns] or [pl.lit(0.0)]
    valid_aa = [c for c in aa_cols if c in df_combined.columns] or [pl.lit(0.0)]
    df_combined = df_combined.with_columns(
        pl.sum_horizontal(valid_ba).alias("bid_vols_10"),
        pl.sum_horizontal(valid_aa).alias("ask_vols_10")
    )
    # [FIX] 提前计算 oir_10，避免 Polars 在后续 alias 时解析异常
    df_combined = df_combined.with_columns(
        ((pl.col("bid_vols_10") - pl.col("ask_vols_10")) / (pl.col("bid_vols_10") + pl.col("ask_vols_10") + 1e-9)).alias("oir_10")
    )
    # 深度集中度: 前3档占全10档的比例
    front_3_bid = [f'申买量{i}' for i in range(1, 4)]
    front_3_ask = [f'申卖量{i}' for i in range(1, 4)]
    valid_f3_bid = [c for c in front_3_bid if c in df_combined.columns] or [pl.lit(0.0)]
    valid_f3_ask = [c for c in front_3_ask if c in df_combined.columns] or [pl.lit(0.0)]
    
    df_combined = df_combined.with_columns(
        ((pl.sum_horizontal(valid_f3_bid) + pl.sum_horizontal(valid_f3_ask)) / 
         (pl.col("bid_vols_10") + pl.col("ask_vols_10") + 1e-9)).alias("depth_concentration")
    )
    
    # [V5] 流动性真空率 (外层挂单多而内层极度吃紧)
    df_combined = df_combined.with_columns(
        (1.0 - pl.col("depth_concentration")).alias("liquidity_void_rate")
    )
    
    mat_bp = df_combined.select(bp_cols).to_numpy() if all(c in df_combined.columns for c in bp_cols) else np.zeros((df_combined.height, 10))
    mat_ba = df_combined.select(ba_cols).to_numpy() if all(c in df_combined.columns for c in ba_cols) else np.zeros((df_combined.height, 10))
    mat_ap = df_combined.select(ap_cols).to_numpy() if all(c in df_combined.columns for c in ap_cols) else np.zeros((df_combined.height, 10))
    mat_aa = df_combined.select(aa_cols).to_numpy() if all(c in df_combined.columns for c in aa_cols) else np.zeros((df_combined.height, 10))
    
    mid_prices = (mat_bp[:, 0] + mat_ap[:, 0]) / 2.0
    ask1_zeros = (mat_ap[:, 0] == 0.0)
    mid_prices[ask1_zeros] = mat_bp[ask1_zeros, 0] 
    
    # 因子计算
    ofi_multi = factor_ofi_multi_level_ashare(mat_bp, mat_ba, mat_ap, mat_aa, levels=10)
    volatility = factor_volatility_ashare(mid_prices, window=20)
    
    if "叫买总量" in df_combined.columns and "叫卖总量" in df_combined.columns:
        total_bids = df_combined.select("叫买总量").to_numpy().flatten()
        total_asks = df_combined.select("叫卖总量").to_numpy().flatten()
    else:
        total_bids = np.zeros(df_combined.height)
        total_asks = np.zeros(df_combined.height)
    obi = factor_simple_obi_ashare(total_bids, total_asks)
    
    price_diff = factor_weighted_price_diff_ashare(mat_bp, mat_ba, mat_ap, mat_aa, levels=5)
    depth_var = factor_depth_variance(mat_ba, mat_aa, levels=10)
    book_slope = factor_book_slope(mat_ba, mat_aa, levels=10)
    obi_mom = factor_obi_momentum_ashare(total_bids, total_asks, window=20)
    liquidity_void_probe = factor_liquidity_void_probe(mat_ba, mat_aa)
    
    # 挂载特征
    # 核心修复: 必须检查并确保所有列都存在，否则赋予默认值 0.0
    cols_to_select = ["万得代码", "datetime", "session_id"]
    for c, alias in [("申卖价1", "ap1"), ("申买价1", "bp1"), ("申买量1", "bv1"), ("申卖量1", "av1")]:
       if c in df_combined.columns:
           cols_to_select.append(pl.col(c).alias(alias))
       else:
           cols_to_select.append(pl.lit(0.0).alias(alias))
           
    cols_to_select.extend(["bid_vols_10", "ask_vols_10", "depth_concentration", "liquidity_void_rate", "oir_10"])
    
    for c in ["pre_close", "last_price", "当日累计成交量", "品种总数"]:
       if c in df_combined.columns:
           cols_to_select.append(pl.col(c))
       else:
           cols_to_select.append(pl.lit(0.0).alias(c))

    df_features = df_combined.select(cols_to_select).with_columns([
        pl.Series("factor_ofi_10", ofi_multi),
        pl.Series("factor_volatility", volatility),
        pl.Series("factor_obi", obi),
        pl.Series("factor_price_diff", price_diff),
        pl.Series("factor_depth_var", depth_var),
        pl.Series("factor_book_slope", book_slope),
        pl.Series("factor_obi_mom", obi_mom),
        pl.Series("factor_liquidity_void_probe", liquidity_void_probe),
        (pl.col("ap1") == 0.0).cast(pl.Int32).alias("is_limit_up"),
        (pl.col("bv1") * pl.col("bp1")).alias("limit_up_order_amt"),
        # 新增微观因子
        ((pl.col("ap1") - pl.col("bp1")) / (pl.col("bp1") + 1e-9)).alias("spread_ratio"),
        ((pl.col("bv1") - pl.col("av1")) / (pl.col("bv1") + pl.col("av1") + 1e-9)).alias("bid_ask_imbalance"),
        # [NEW Advanced L2] 
        # 1. 加权中间价偏离度 (WMP Divergence)
        ((((pl.col("bp1") * pl.col("av1") + pl.col("ap1") * pl.col("bv1")) / (pl.col("bv1") + pl.col("av1") + 1e-9)) / (pl.col("last_price") + 1e-9)) - 1.0).alias("wmp_divergence"),
        # (OIR_10 已经在更早阶段计算并纳入了)
        
        # [V9] 涨停特色微观因子 2: 隔夜溢价排单代理 (Overnight Queuing Proxy)
        (pl.when(pl.col("session_id") == 0)
           .then(pl.col("bv1") / (pl.col("bid_vols_10") + 1e-9))
           .otherwise(0.0)).alias("overnight_queuing_proxy"),
           
        # [V9] 涨停特色微观因子 3: 涨停磁吸效应 (Limit-up Magnet Effect)
        (pl.when((pl.col("ap1") > 0.0) & (pl.col("last_price") / (pl.col("pre_close") + 1e-9) > 1.08))
           .then((pl.col("bv1") - pl.col("av1")) / (pl.col("bid_vols_10") + pl.col("ask_vols_10") + 1e-9))
           .otherwise(0.0)).alias("limit_up_magnet_effect")
    ])
    
    # === [V10.2] Depth Weighted Price Impact (深度加权冲击成本) ===
    # 模拟吃掉卖盘全部 10 档后的加权均价 vs 买一价的偏离
    depth_impact = factor_depth_weighted_impact_ashare(mat_ap, mat_aa)
    df_features = df_features.with_columns(pl.Series("depth_weighted_impact", depth_impact))
    
    # 时间聚合 (1分钟)
    df_agg = (
        df_features
        .with_columns(pl.col("datetime").dt.truncate(f"{time_step_seconds}s").alias("period_start"))
        .group_by(["万得代码", "period_start"])
        .agg([
            pl.col("bp1").last().alias("close_bp1"),
            pl.col("ap1").last().alias("close_ap1"),
            pl.col("pre_close").last().alias("pre_close"),
            pl.col("last_price").last().alias("last_price"),
            pl.col("session_id").last().alias("session_id"),
            pl.col("factor_ofi_10").sum().alias("ofi_sum"),
            pl.col("factor_volatility").mean().alias("volatility_mean"),
            pl.col("factor_obi").last().alias("obi_last"),
            pl.col("factor_price_diff").mean().alias("price_diff_mean"),
            pl.col("factor_depth_var").last().alias("depth_var_last"),
            pl.col("spread_ratio").mean().alias("spread_ratio_mean"),
            pl.col("bid_ask_imbalance").mean().alias("bid_ask_imbalance_mean"),
            pl.col("wmp_divergence").mean().alias("wmp_divergence_mean"),
            pl.col("oir_10").mean().alias("oir_10_mean"),
            pl.col("factor_book_slope").mean().alias("book_slope_mean"),
            pl.col("depth_concentration").mean().alias("depth_concentration_mean"),
            pl.col("factor_obi_mom").mean().alias("obi_momentum"),
            pl.col("factor_liquidity_void_probe").mean().alias("liquidity_void_probe"),
            pl.col("liquidity_void_rate").mean().alias("liquidity_void_rate_mean"),
            
            # [BugFix] V9 涨停特色微观博弈因子缺失聚合恢复
            pl.col("overnight_queuing_proxy").mean().alias("overnight_queuing_proxy"),
            pl.col("limit_up_magnet_effect").max().alias("limit_up_magnet_effect"),
            
            # [V5] 冰山撤补代理 (Iceberg Depth Replenishment): 盘口挂单剧震但是价格却波澜不惊
            (((pl.col("bid_vols_10") + pl.col("ask_vols_10")).std().fill_null(0.0) / (pl.col("bid_vols_10") + pl.col("ask_vols_10")).mean().fill_null(1e-9)) / 
            ((pl.col("bp1").std().fill_null(0.0) / (pl.col("bp1").mean().fill_null(1e-9))) + 1e-6)).alias("iceberg_replenish_proxy"),
            
            # [V10] 冰山极速动量 (Iceberg Momentum): 计算本分钟挂单总量与上分钟剧烈变化
            ((pl.col("bid_vols_10") + pl.col("ask_vols_10")).last() / ((pl.col("bid_vols_10") + pl.col("ask_vols_10")).first() + 1e-9)).alias("iceberg_momentum"),
            
            # [V10] 换手率/波动率效率比 (Volume-Volatility Efficiency)
            # 单位波动需要的成交量推动（换手越高效说明筹码越扎实）
            ((pl.col("当日累计成交量").last() - pl.col("当日累计成交量").first()) / (pl.col("factor_volatility").mean() + 1e-9)).alias("vol_volatility_efficiency"),
            
            # [V10.2] 深度加权冲击成本 (Depth Weighted Impact)
            pl.col("depth_weighted_impact").mean().alias("depth_weighted_impact_mean"),
            
            # [V10.2] 盘口弹性代理 (LOB Resilience): spread 波动率的倒数
            # spread_ratio std 越小 = 盘口越稳 = 弹性越大
            (1.0 / (pl.col("spread_ratio").std().fill_null(1e-6) + 1e-6)).alias("lob_resilience"),
            
            # === [V10.2] 冰山/隐藏/拆分单检测因子 ===
            
            # 1. Depth Replenish Speed (盘口回填速度): 
            # 最低深度 vs 末端深度 → 回填越快 = 冰山单越大
            ((pl.col("bid_vols_10") + pl.col("ask_vols_10")).last() / 
             ((pl.col("bid_vols_10") + pl.col("ask_vols_10")).min() + 1e-9)).alias("depth_replenish_speed"),
            
            # 2. Stealth Accumulation (隐形吸筹指数):
            # OFI 净量大但价格几乎不动 → 有人用冰山单偷偷堆仓
            (pl.col("factor_ofi_10").sum().abs() / 
             ((pl.col("last_price").last() - pl.col("last_price").first()).abs() + 1e-9)).alias("stealth_accumulation"),
            
            # 3. Phantom Liquidity (幽灵流动性):
            # 盘口深度剧烈波动但成交量未同步 → 闪挂闪撤的诱导单
            ((pl.col("bid_vols_10") + pl.col("ask_vols_10")).std().fill_null(0.0) / 
             ((pl.col("当日累计成交量").last() - pl.col("当日累计成交量").first()).abs() + 1e-9)).alias("phantom_liquidity"),
            
            # 4. Hidden Fill Ratio (暗单成交占比):
            # 实际成交量 / 可见最大深度 → 超过1说明有隐藏订单在暗中填补
            ((pl.col("当日累计成交量").last() - pl.col("当日累计成交量").first()) / 
             ((pl.col("bid_vols_10") + pl.col("ask_vols_10")).max() + 1e-9)).alias("hidden_fill_ratio"),
            
            # 5. Depth Asymmetry Persistence (深度不对称持久度):
            # 买卖深度差的符号一致性 → 持续单边厚 = 机构维护
            ((pl.col("bid_vols_10") - pl.col("ask_vols_10")).mean() / 
             ((pl.col("bid_vols_10") - pl.col("ask_vols_10")).std().fill_null(1e-6) + 1e-6)).alias("depth_asym_persistence"),
             
            # === [V10.2] A股打板巅峰专项因子 (Snapshot 侧) ===
            
            # 1. 订单簿买卖斜率差 (Bid-Ask Slope Diff) 
            # (由于没有逐档数据存储，用深度比模拟) -> 买单集中在前3档而卖单均匀=主力托盘不攻
            ((pl.col("depth_concentration") - (pl.col("ask_vols_10") / (pl.col("bid_vols_10") + pl.col("ask_vols_10") + 1e-9)))).mean().alias("bid_ask_slope_diff"),
            
            # 2. 盘口失衡凸性 (LOB Convexity)
            # 失衡度的二阶变化率，捕捉卖压耗尽瞬间
            (pl.col("bid_ask_imbalance") - pl.col("bid_ask_imbalance").shift(1)).std().fill_null(0.0).alias("lob_convexity"),
            
            # 3. 封板消耗速率 (Limit-Up Consumption Rate)
            # 距离涨停极近时，卖盘（挂单）减少的最高速度
            pl.when((pl.col("ap1") > 0.0) & (pl.col("ap1") / (pl.col("pre_close") + 1e-9) > 1.09))
              .then((pl.col("ask_vols_10").shift(1) - pl.col("ask_vols_10"))).otherwise(0.0).max().alias("limit_up_consumption_rate"),
              
            # 4. 近板抢筹烈度 (Near Limit-Up Accumulation)
            # 距离涨停 <2% 时的平均 OFI 净买入，极高表示资金点火意愿强烈
            pl.when((pl.col("last_price") / (pl.col("pre_close") + 1e-9) > 1.08) & ((pl.col("ap1") > 0.0) | (pl.col("bp1") > 0.0)))
              .then(pl.col("factor_ofi_10")).otherwise(0.0).sum().alias("near_limit_up_accumulation"),
            
            pl.col("当日累计成交量").last().alias("cum_volume"),
            pl.col("is_limit_up").max().alias("touched_limit_up"),
            pl.col("limit_up_order_amt").max().alias("max_limit_up_order_amt"),
            
            # === [NEW] Snapshot 微观换手率 (Micro Turnover Proxies) ===
            # 1. 盘口换手率(Order Book Turnover): 本分钟真实成交量增量与平均挂单厚度的比例
            # 极高的盘口换手率意味着当前盘口正在经历激烈的真实筹码交换
            ((pl.col("当日累计成交量").last() - pl.col("当日累计成交量").first()) / 
             ((pl.col("bid_vols_10") + pl.col("ask_vols_10")).mean() + 1e-9)).alias("order_book_turnover"),
             
            # 2. 挂单周转率(Visible Supply Turnover): 按瞬时脉冲放量，需要消耗几次当前可视的最大深度的流动性池
            # 越大说明主力用成交量打穿盘口的暴力意愿越强
            ((pl.col("当日累计成交量").last() - pl.col("当日累计成交量").first()) / 
             ((pl.col("bid_vols_10") + pl.col("ask_vols_10")).max() + 1e-9)).alias("visible_supply_turnover"),

            pl.len().alias("tick_count")
        ])
    )
    return df_agg

def compute_market_level_factors(df_agg: pl.DataFrame) -> pl.DataFrame:
    """第二阶段：计算全市场情绪因子并合并"""
    if df_agg.height == 0:
        return df_agg
        
    df_market = df_agg.group_by("period_start").agg([
        pl.col("ofi_sum").sum().alias("market_ofi_sum"),
        pl.col("touched_limit_up").sum().alias("market_limit_up_count")
    ])
    
    df_final = df_agg.join(df_market, on="period_start", how="left")
    
    df_final = (
        df_final.with_columns(
            ((pl.col("last_price") / pl.col("pre_close")) - 1).alias("return_from_pre_close")
        )
        .fill_nan(0.0)
        .sort(["period_start", "万得代码"])
    )
    return df_final

def compute_and_aggregate_ashare(df_combined: pl.DataFrame, time_step_seconds: int = 60) -> pl.DataFrame:
    """兼容旧接口的封装"""
    df_agg = compute_stock_level_factors(df_combined, time_step_seconds)
    return compute_market_level_factors(df_agg)
