import polars as pl
import numpy as np
import os

def compute_tick_factors_for_date(date_str: str, base_dir: str = None) -> pl.DataFrame:
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "l2_tick")
        
    """
    [V10 Optimized] 从 l2_tick 逐笔成交数据中提取因子。逐文件处理，防止内存崩溃。
    """
    import shutil
    import gc

    date_dir = os.path.join(base_dir, f"date={date_str}")
    if not os.path.isdir(date_dir):
        return pl.DataFrame()
    
    parquet_files = [os.path.join(date_dir, f) for f in os.listdir(date_dir) if f.endswith(".parquet")]
    if not parquet_files:
        return pl.DataFrame()
    
    print(f"    [Tick] 正在处理 {len(parquet_files)} 个逐笔成交文件...")
    
    # 临时缓冲目录，直接强制放去大容量 E 盘以防爆盘，并使用 uuid 防止 Windows rmtree 延迟删除竞态导致 OsError 3
    import uuid
    temp_tick_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", f"temp_tick_{date_str}_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_tick_dir, exist_ok=True)
    
    needed = ["万得代码", "自然日", "时间", "成交价格", "成交数量", "委托代码"]
    
    def process_single_tick_file(args):
        idx, f = args
        if not os.path.isfile(f):
            print(f"    [Skip] Tick 文件不存在: {f}")
            return False
        try:
            df = pl.read_parquet(f, low_memory=True)
            available = [c for c in needed if c in df.columns]
            if "自然日" not in available or "时间" not in available:
                return False
                
            df = df.select(available)
            
            # 过滤无效行
            df = df.filter(pl.col("自然日").cast(pl.Int64, strict=False) > 20000000)
            if df.height == 0:
                return False
                
            # 快速时间转换
            df = df.with_columns(
                ((pl.col("自然日").cast(pl.Int64) * 1000000000) + pl.col("时间").cast(pl.Int64)).alias("raw_ts")
            )
            df = df.with_columns(
                pl.col("raw_ts").cast(pl.Utf8).str.strptime(pl.Datetime, format="%Y%m%d%H%M%S%3f", strict=False).alias("datetime")
            ).drop("raw_ts")
            
            df = df.filter(pl.col("datetime").is_not_null())
            if df.height == 0:
                return False
            
            # 基础计算
            df = df.with_columns(
                (pl.col("成交价格") / 10000.0).alias("tick_price"),
                (pl.col("成交价格") * pl.col("成交数量") / 10000.0).alias("trade_amt"),
                pl.col("成交数量").alias("tick_vol"),
                pl.col("委托代码").cast(pl.Utf8, strict=False) # 安全转型防止部分Parquet被推断为i32报错
            )
            
            # 因子逻辑
            BIG = 500000.0
            HUGE = 2000000.0
            df = df.with_columns(
                pl.when(pl.col("委托代码") == "B").then(pl.col("trade_amt")).otherwise(0.0).alias("active_buy_amt"),
                pl.when(pl.col("委托代码") == "S").then(pl.col("trade_amt")).otherwise(0.0).alias("active_sell_amt"),
                pl.when((pl.col("委托代码") == "B") & (pl.col("trade_amt") >= BIG)).then(pl.col("trade_amt")).otherwise(0.0).alias("big_buy_amt"),
                pl.when((pl.col("委托代码") == "S") & (pl.col("trade_amt") >= BIG)).then(pl.col("trade_amt")).otherwise(0.0).alias("big_sell_amt"),
                pl.when((pl.col("委托代码") == "B") & (pl.col("trade_amt") >= HUGE)).then(pl.col("trade_amt")).otherwise(0.0).alias("huge_buy_amt"),
                pl.when((pl.col("委托代码") == "S") & (pl.col("trade_amt") >= HUGE)).then(pl.col("trade_amt")).otherwise(0.0).alias("huge_sell_amt"),
                # [V9] 新增撤单统计 (Cancel Orders)，通常部分交易所L2中 C代表撤单 (Cancel)
                pl.when(pl.col("委托代码") == "C").then(pl.col("trade_amt")).otherwise(0.0).alias("cancel_amt"),
                pl.when((pl.col("委托代码") == "C") & (pl.col("trade_amt") >= BIG)).then(pl.col("trade_amt")).otherwise(0.0).alias("big_cancel_amt")
            )
            
            # 分钟聚合
            df_agg = (
                df.with_columns(pl.col("datetime").dt.truncate("60s").alias("period_start"))
                .group_by(["万得代码", "period_start"])
                .agg([
                    pl.len().alias("tick_count_raw"),
                    pl.col("active_buy_amt").sum().alias("active_buy_amt"),
                    pl.col("active_sell_amt").sum().alias("active_sell_amt"),
                    pl.col("big_buy_amt").sum().alias("big_buy_amt"),
                    pl.col("big_sell_amt").sum().alias("big_sell_amt"),
                    pl.col("huge_buy_amt").sum().alias("huge_buy_amt"),
                    pl.col("huge_sell_amt").sum().alias("huge_sell_amt"),
                    pl.col("cancel_amt").sum().alias("cancel_amt"),
                    pl.col("big_cancel_amt").sum().alias("big_cancel_amt"),
                    pl.col("trade_amt").sum().alias("total_trade_amt"),
                    pl.col("tick_vol").sum().alias("total_trade_vol"),
                    pl.col("tick_price").last().alias("period_last_price"),
                    pl.col("tick_price").first().alias("period_first_price"),
                    pl.col("tick_price").max().alias("period_high_price"),
                    pl.col("tick_price").min().alias("period_low_price"),
                    ((pl.col("tick_price") / pl.col("tick_price").shift(1) - 1).fill_null(0.0).pow(2)).sum().alias("tick_rv"),
                    # [V10.2] 成交到达率比 (Trade Arrival Rate Ratio): 主买笔数/主卖笔数
                    pl.when(pl.col("委托代码") == "B").then(1).otherwise(0).sum().alias("buy_tick_count"),
                    pl.when(pl.col("委托代码") == "S").then(1).otherwise(0).sum().alias("sell_tick_count"),
                    # [V10.2] HHI 成交规模集中度: sum((trade_i / total)^2)
                    (pl.col("trade_amt").pow(2)).sum().alias("trade_amt_sq_sum"),
                    # [V10.2] 扰单穿透强度 (Sweep): 单笔最大成交量
                    pl.col("trade_amt").max().alias("max_single_trade_amt"),
                    # [V11 防守因子 1] 尾部风险惩戒: 下滑放量/反弹缩量
                    pl.when(pl.col("tick_price") < pl.col("tick_price").shift(1)).then(pl.col("tick_vol")).otherwise(0.0).sum().alias("down_vol_sum"),
                    pl.when(pl.col("tick_price") > pl.col("tick_price").shift(1)).then(pl.col("tick_vol")).otherwise(0.0).sum().alias("up_vol_sum"),
                    # [V11 防守因子 2] 吃档耗时代理: 平均活跃买单笔数与金额的比例
                    (pl.col("active_buy_amt").sum() / (pl.when(pl.col("委托代码") == "B").then(1).otherwise(0).sum() + 1e-9)).alias("avg_active_buy_size"),
                    # [V10.2 大打板因子 Volume Profile]
                    # 计算本分钟内的 vwap: SUM(金额)/SUM(成交量)，这里先存金额和总量
                    # idx_max_trade 是一个Aggregate内计算的, 没法用来点查 tick_price
                    pl.col("tick_price").arg_max().alias("idx_max_trade"),
                    # [V10.2 冰山拆单]
                    (pl.col("datetime").cast(pl.Int64).diff().std().fill_null(0.0) / 1000000.0).alias("trade_interval_std"),
                    pl.when(pl.col("tick_vol") == pl.col("tick_vol").shift(1)).then(1).otherwise(0).sum().alias("same_size_count"),
                    pl.col("tick_vol").n_unique().alias("unique_size_count"),
                    # [V10.2 涨停自激代理 (霍克斯自激)]
                    # 连续产生两笔主动买入的时间间隔的倒数(极速追单) -> 这里用主动买入的总金额密度作为 proxy
                    pl.col("active_buy_amt").sum().alias("self_excitation_proxy_amt")
                ])
            )
            
            df_agg = df_agg.rename({
                "active_buy_amt": "active_buy_total",
                "active_sell_amt": "active_sell_total",
                "big_buy_amt": "big_buy_total",
                "big_sell_amt": "big_sell_total",
                "huge_buy_amt": "huge_buy_total",
                "huge_sell_amt": "huge_sell_total",
                "cancel_amt": "cancel_total",
                "big_cancel_amt": "big_cancel_total"
            })
            
            # [V10.2] 计算单文件级的大单后反转率 (Large Trade Reversal)
            # 最大笔交易的价格 vs 本分钟末价的方向
            # idx_max_trade 是一个Aggregate内计算的, 没法用来点查 tick_price
            # 改为简易代理: sign(period_last - period_first) * (-1 if max trade was buy else 1)
            if "period_last_price" in df_agg.columns and "period_first_price" in df_agg.columns:
                df_agg = df_agg.with_columns(
                    pl.when(pl.col("period_last_price") < pl.col("period_first_price"))
                    .then(pl.lit(1.0))  # 价格在大单后下跌 = 反转信号
                    .otherwise(pl.lit(0.0)).alias("large_trade_reversal_flag")
                )
                
            # [V10.2 Volume Profile] 单分钟内的 VWAP
            if "total_trade_amt" in df_agg.columns and "total_trade_vol" in df_agg.columns:
                df_agg = df_agg.with_columns(
                    (pl.col("total_trade_amt") / (pl.col("total_trade_vol") + 1e-9)).alias("minute_vwap")
                )
            
            if df_agg.height > 0:
                df_agg.write_parquet(os.path.join(temp_tick_dir, f"tick_part_{idx}.parquet"))
                del df, df_agg
                return True
            else:
                del df, df_agg
                return False
        except (MemoryError, Exception) as e:
            print(f"    [Tick Error] {f}: {e}")
            # OOM 兜底：强制回收已分配的内存碎片
            import gc
            gc.collect()
            return False

    # [V13 OOM Fix] 串行处理，避免多线程同时驻留多份完整 Tick 数据导致内存峰值爆炸
    results = []
    for i, f in enumerate(parquet_files):
        ok = process_single_tick_file((i, f))
        results.append(ok)
        if (i + 1) % 10 == 0:
            gc.collect()
            print(f"    [Tick] 已处理 {i+1}/{len(parquet_files)} 个文件...")
    
    count = sum(results)
    gc.collect()
            
    if count == 0:
        print(f"    [Tick] {date_str} 无有效聚合结果")
        try:
            shutil.rmtree(temp_tick_dir)
        except:
            pass
        return pl.DataFrame()
        
    # 合并聚合后的 Tick 因子
    df_all_tick = pl.read_parquet(os.path.join(temp_tick_dir, "*.parquet"), low_memory=True)
    try:
        shutil.rmtree(temp_tick_dir)
    except:
        pass
    
    # 最终汇总
    df_all_tick = df_all_tick.group_by(["万得代码", "period_start"]).agg([
            pl.col("tick_count_raw").sum().alias("tick_count_raw"),
        pl.col("active_buy_total").sum().alias("active_buy_total"),
        pl.col("active_sell_total").sum().alias("active_sell_total"),
        pl.col("big_buy_total").sum().alias("big_buy_total"),
        pl.col("big_sell_total").sum().alias("big_sell_total"),
        pl.col("huge_buy_total").sum().alias("huge_buy_total"),
        pl.col("huge_sell_total").sum().alias("huge_sell_total"),
        pl.col("cancel_total").sum().alias("cancel_total"),
        pl.col("big_cancel_total").sum().alias("big_cancel_total"),
        # [V10] 新增主买方/主卖方的撤单动量追踪
        pl.when(pl.col("active_buy_total") > pl.col("active_sell_total"))
          .then(pl.col("cancel_total"))
          .otherwise(0.0).sum().alias("buy_side_cancel_amt"),
        pl.when(pl.col("active_sell_total") > pl.col("active_buy_total"))
          .then(pl.col("cancel_total"))
          .otherwise(0.0).sum().alias("sell_side_cancel_amt"),
          
        pl.col("total_trade_amt").sum().alias("total_trade_amt"),
        pl.col("total_trade_vol").sum().alias("total_trade_vol"),
        pl.col("period_last_price").last().alias("period_last_price"),
        pl.col("period_first_price").first().alias("period_first_price"),
        pl.col("period_high_price").max().alias("period_high_price"),
        pl.col("period_low_price").min().alias("period_low_price"),
        pl.col("tick_rv").sum().alias("tick_rv"),
        # [V10.2] 新增 Tick 侧因子原始聚合
        pl.col("buy_tick_count").sum().alias("buy_tick_count"),
        pl.col("sell_tick_count").sum().alias("sell_tick_count"),
        pl.col("trade_amt_sq_sum").sum().alias("trade_amt_sq_sum"),
        pl.col("max_single_trade_amt").max().alias("max_single_trade_amt"),
        pl.col("large_trade_reversal_flag").mean().alias("large_trade_reversal_flag"),
        # [V10.2 冰山拆单]
        pl.col("trade_interval_std").mean().alias("trade_interval_std"),
        pl.col("same_size_count").sum().alias("same_size_count"),
        pl.col("unique_size_count").sum().alias("unique_size_count"),
        # [V10.2 Volume Profile & Self-Excitation]
        pl.col("minute_vwap").mean().alias("minute_vwap"),
        pl.col("self_excitation_proxy_amt").sum().alias("self_excitation_proxy_amt"),
        # [V11] 临时传出的数据
        pl.col("down_vol_sum").sum().alias("down_vol_sum"),
        pl.col("up_vol_sum").sum().alias("up_vol_sum"),
        pl.col("avg_active_buy_size").mean().alias("avg_active_buy_size")
    ])
    
    print(f"    [Tick] {date_str} 聚合完成，截面行数: {df_all_tick.height}")
    df_all_tick = df_all_tick.with_columns(
        (pl.col("down_vol_sum") / (pl.col("up_vol_sum") + 1e-9)).alias("tail_risk_skew"),
        (pl.col("trade_interval_std") * 1000.0 / (pl.col("avg_active_buy_size") + 1e-9)).alias("spread_crossing_speed"),
        
        (pl.col("active_buy_total") / (pl.col("active_buy_total") + pl.col("active_sell_total") + 1e-9)).alias("active_buy_ratio"),
        (pl.col("big_buy_total") - pl.col("big_sell_total")).alias("big_net_inflow"),
        (pl.col("huge_buy_total") - pl.col("huge_sell_total")).alias("huge_net_inflow"),
        ((pl.col("active_buy_total") + pl.col("active_sell_total")) / (pl.col("tick_count_raw") + 1e-9)).alias("avg_trade_size"),
        (pl.col("tick_count_raw") / 60.0).alias("tick_freq_per_sec"),
        ((pl.col("period_last_price") - (pl.col("total_trade_amt") / (pl.col("total_trade_vol") + 1e-9))) / 
         (pl.col("total_trade_amt") / (pl.col("total_trade_vol") + 1e-9) + 1e-9)).alias("vwap_divergence"),
        # [V6] 1. VPIN (知情交易概率)
        ((pl.col("active_buy_total") - pl.col("active_sell_total")).abs() / 
         (pl.col("total_trade_vol") + 1e-9)).alias("vpin"),
        ((pl.col("period_last_price") - pl.col("period_first_price")) / (pl.col("active_buy_total") - pl.col("active_sell_total") + 1e-9)).alias("kyle_lambda"),
        (((pl.col("period_last_price") - pl.col("period_first_price")) / (pl.col("period_first_price") + 1e-9)).abs() / (pl.col("total_trade_amt") + 1e-9) * 1e8).alias("amihud_illiq"),
        # [V6] 4. Trade Entropy (资金熵)
        (pl.col("tick_rv") * pl.col("tick_count_raw").log1p()).alias("trade_entropy"),
        # [V6] 4. Smart Money PSI
        ((pl.col("big_buy_total") - pl.col("big_sell_total")) / (pl.col("total_trade_amt") + 1e-9)).alias("smart_psi"),
        # [V6] 5. Path of Least Resistance (阻力最小路径)
        (((pl.col("period_last_price") - pl.col("period_first_price")) / (pl.col("period_first_price") + 1e-9)) / (pl.col("total_trade_vol") + 1e-9)).alias("least_resistance_path"),
        
        # [V7] SMC 1: K线实体吞没率 (Displacement Imbalance Rate)
        ((pl.col("period_last_price") - pl.col("period_first_price")).abs() / 
         ((pl.col("period_high_price") - pl.col("period_low_price")) + 1e-9)).alias("smc_imbalance_rate"),
         
        # [V7] SMC 2: 供需区块强度 (Supply/Demand Block Proxy)
        # 用方向符号(1/-1) * 吞没率 * 总成交量
        (pl.when(pl.col("period_last_price") > pl.col("period_first_price")).then(pl.lit(1)).otherwise(pl.lit(-1)) * 
         ((pl.col("period_last_price") - pl.col("period_first_price")).abs() / ((pl.col("period_high_price") - pl.col("period_low_price")) + 1e-9)) * 
         pl.col("total_trade_amt")).alias("smc_supply_demand_proxy"),
         
        # [V9] 涨停特色微观因子 1: 极端撤单占比 (Cancel Ratio Extreme)
        # 撤单金额占总流动性(成交+撤单)的比重，捕捉虚假大单诱多拆单行为
        (pl.col("cancel_total") / (pl.col("total_trade_amt") + pl.col("cancel_total") + 1e-9)).alias("cancel_ratio_extreme"),
        
        # [V10] 虚假托单欺骗性偏移 (Cancel Skewness) -> 捕捉 Spoofing
        ((pl.col("buy_side_cancel_amt") - pl.col("sell_side_cancel_amt")) / (pl.col("cancel_total") + 1e-9)).alias("cancel_skewness"),
        
        # === [V10.2] 新增 7 个订单流因子 ===
        # 1. 成交到达率比 (Trade Arrival Rate Ratio)
        (pl.col("buy_tick_count") / (pl.col("sell_tick_count") + 1e-9)).alias("trade_arrival_ratio"),
        
        # 2. 成交规模集中度 (HHI)
        (pl.col("trade_amt_sq_sum") / (pl.col("total_trade_amt").pow(2) + 1e-9)).alias("trade_size_hhi"),
        
        # 3. 扫单穿透强度 (Sweep Intensity): 单笔最大成交 / 平均成交
        (pl.col("max_single_trade_amt") / ((pl.col("total_trade_amt") / (pl.col("tick_count_raw") + 1e-9)) + 1e-9)).alias("sweep_intensity"),
        
        # 4. 大单后反转率 (Large Trade Reversal)
        pl.col("large_trade_reversal_flag").alias("large_trade_reversal"),
        
        # 5. 已实现价差代理 (Realized Spread Proxy)
        ((pl.col("active_buy_total") - pl.col("active_sell_total")) / (pl.col("total_trade_amt") + 1e-9) *
         (pl.col("period_last_price") - pl.col("period_first_price")) / (pl.col("period_first_price") + 1e-9)).alias("realized_spread_proxy"),
        
        # 6. 量桶 VPIN (Volume-Bucketed VPIN)
        ((pl.col("active_buy_total") - pl.col("active_sell_total")).abs() /
         (pl.col("total_trade_vol") + 1e-9) * (pl.col("tick_count_raw") / 60.0).log1p()).alias("vpin_vol_bucket"),
         
        # === [V10.2] 冰山/隐藏/拆分成单 ===
        # 1. 连续同量比 (Consecutive Same-Size Ratio)
        (pl.col("same_size_count") / (pl.col("tick_count_raw") + 1e-9)).alias("consecutive_same_size_ratio"),
        
        # 2. 成交间隔规律性 (Trade Interval Regularity): 标准差的倒数，越大越规律
        (pl.lit(1.0) / (pl.col("trade_interval_std") + 1e-6)).alias("trade_interval_regularity"),
        
        # 3. 规模聚类度 (Trade Size Clustering): 1 - (独有size数 / 总tick数)
        (pl.lit(1.0) - pl.col("unique_size_count") / (pl.col("tick_count_raw") + 1e-9)).alias("trade_size_clustering"),
        
        # === [V10.2] Volume Profile & 涨停自激 (Tick 侧) ===
        
        # 1. POC 中枢偏离度 (POC Distance): 当前价大幅偏离无量VWAP极易回落
        ((pl.col("period_last_price") - pl.col("minute_vwap")) / (pl.col("minute_vwap") + 1e-9)).alias("poc_distance"),
        
        # 2. 价值区间突破力 (Value Area Breakout): 收盘显著高于均价的偏离度
        ((pl.col("period_last_price") - pl.col("minute_vwap")) / 
         ((pl.col("period_high_price") - pl.col("period_low_price")) + 1e-9)).alias("value_area_breakout"),
         
        # 3. 量价分布偏度 (VP Skewness): 筹码偏上(分布负偏)还是偏下(正偏)
        ((pl.col("minute_vwap") - ((pl.col("period_high_price") + pl.col("period_low_price")) / 2.0)) / 
         ((pl.col("period_high_price") - pl.col("period_low_price")) + 1e-9)).alias("vp_skewness"),
         
        # 4. 霍克斯自激指数代理 (Hawkes Self-Excitation Proxy): 集中大额主买密度
        (pl.col("self_excitation_proxy_amt") / (pl.col("tick_count_raw") * pl.col("trade_interval_std").fill_null(1.0) + 1e-9)).alias("self_excitation_proxy")
    )
    
    # [V6 & V7] 添加时序依赖因子 (Kaufman ER, CVD Divergence, FVG Gap)
    df_all_tick = df_all_tick.sort(["万得代码", "period_start"])
    df_all_tick = df_all_tick.with_columns(
        (pl.col("active_buy_total") - pl.col("active_sell_total")).alias("volume_delta")
    ).with_columns(
        pl.col("volume_delta").cum_sum().over("万得代码").alias("cvd"),
        pl.col("total_trade_vol").cum_sum().over("万得代码").alias("cum_vol"),
        pl.col("period_first_price").first().over("万得代码").alias("day_open_price")
    ).with_columns(
        # [V6] 3. CVD Divergence (累计量差背离) = 累计主动量比 - 盘中涨幅
        ((pl.col("cvd") / (pl.col("cum_vol") + 1e-9)) - 
         ((pl.col("period_last_price") - pl.col("day_open_price")) / (pl.col("day_open_price") + 1e-9))).alias("cvd_divergence"),
         
        # === [NEW] Tick 微观换手率 (Micro Turnover Proxies) ===
        # 1. 日内资金参与度换手 (Intraday Volume Participation): 本分钟成交量占日内总成交比例
        (pl.col("total_trade_vol") / (pl.col("cum_vol") + 1e-9)).alias("intraday_vol_participation"),
        
        # 2. 主动买卖微观换手 (Active Micro Turnover): 主买主卖量均摊到每笔平均单子上的浓度
        ((pl.col("active_buy_total") + pl.col("active_sell_total")) / 
         (pl.col("tick_count_raw") * pl.col("total_trade_vol") / (pl.col("tick_count_raw") + 1e-9) + 1e-9)).alias("active_micro_turnover"),
        
        # [V6] 2. Kaufman Efficiency Ratio (考夫曼效率比)
        ((pl.col("period_last_price") - pl.col("period_last_price").shift(10).over("万得代码")).abs() / 
         (pl.col("period_last_price").diff().abs().rolling_sum(window_size=10).over("万得代码") + 1e-9)).alias("kaufman_er"),
         
        # [V7] SMC 3: Fair Value Gap (FVG)
        # Bullish FVG: Low_t - High_{t-2} > 0
        pl.when(pl.col("period_low_price") > pl.col("period_high_price").shift(2).over("万得代码"))
          .then(pl.col("period_low_price") - pl.col("period_high_price").shift(2).over("万得代码"))
          .otherwise(0.0).alias("smc_fvg_bullish"),
        # Bearish FVG: Low_{t-2} - High_t > 0
        pl.when(pl.col("period_low_price").shift(2).over("万得代码") > pl.col("period_high_price"))
          .then(pl.col("period_low_price").shift(2).over("万得代码") - pl.col("period_high_price"))
          .otherwise(0.0).alias("smc_fvg_bearish"),
          
        # === [V10.2] 9 个 SMC 聪明钱高阶因子 ===
        
        # 1. Institutional Candle Ratio (机构K线率): 实体/全幅, 越大=机构越坚决
        ((pl.col("period_last_price") - pl.col("period_first_price")).abs() /
         ((pl.col("period_high_price") - pl.col("period_low_price")) + 1e-9)).alias("smc_institutional_candle"),
         
        # 2. Displacement Velocity (置换速度): 单位时间的价格变动幅度
        ((pl.col("period_last_price") - pl.col("period_first_price")) /
         (pl.col("period_first_price") + 1e-9)).alias("smc_displacement_velocity"),
         
        # 3. Premium/Discount Zone: 当前价在日内波幅中的位置 (0=折价底部, 1=溢价顶部)
        ((pl.col("period_last_price") - pl.col("period_low_price").rolling_min(window_size=30).over("万得代码")) /
         ((pl.col("period_high_price").rolling_max(window_size=30).over("万得代码") -
           pl.col("period_low_price").rolling_min(window_size=30).over("万得代码")) + 1e-9)).alias("smc_premium_discount"),
         
        # 4. Liquidity Sweep (流动性猎杀): 价格穿越前高后回落 = 扫损信号
        # 上扫: 当前高点 > 前一分钟高点 且 当前收盘 < 前一分钟高点
        pl.when(
            (pl.col("period_high_price") > pl.col("period_high_price").shift(1).over("万得代码")) &
            (pl.col("period_last_price") < pl.col("period_high_price").shift(1).over("万得代码"))
        ).then(pl.lit(1.0))
        # 下扫: 当前低点 < 前一分钟低点 且 当前收盘 > 前一分钟低点
        .when(
            (pl.col("period_low_price") < pl.col("period_low_price").shift(1).over("万得代码")) &
            (pl.col("period_last_price") > pl.col("period_low_price").shift(1).over("万得代码"))
        ).then(pl.lit(-1.0))
        .otherwise(pl.lit(0.0)).alias("smc_liquidity_sweep"),
        
        # 5. CHoCH (Change of Character 变盘信号):
        # 上升趋势中首次出现更低的低点 = 看空变盘
        # 下降趋势中首次出现更高的高点 = 看多变盘
        pl.when(
            (pl.col("period_last_price") > pl.col("period_first_price")) &  # 当前阳线 (上升趋势体感)
            (pl.col("period_low_price") < pl.col("period_low_price").shift(1).over("万得代码"))  # 但低点破前低
        ).then(pl.lit(-1.0))  # 看空变盘
        .when(
            (pl.col("period_last_price") < pl.col("period_first_price")) &  # 当前阴线 (下降趋势体感)
            (pl.col("period_high_price") > pl.col("period_high_price").shift(1).over("万得代码"))  # 但高点破前高
        ).then(pl.lit(1.0))  # 看多变盘
        .otherwise(pl.lit(0.0)).alias("smc_choch"),
        
        # 6. BOS (Break of Structure 结构延续):
        # 多头: 连续创更高的高点
        # 空头: 连续创更低的低点
        pl.when(
            (pl.col("period_high_price") > pl.col("period_high_price").shift(1).over("万得代码")) &
            (pl.col("period_low_price") >= pl.col("period_low_price").shift(1).over("万得代码"))
        ).then(pl.lit(1.0))  # 多头延续
        .when(
            (pl.col("period_low_price") < pl.col("period_low_price").shift(1).over("万得代码")) &
            (pl.col("period_high_price") <= pl.col("period_high_price").shift(1).over("万得代码"))
        ).then(pl.lit(-1.0))  # 空头延续
        .otherwise(pl.lit(0.0)).alias("smc_bos"),
        
        # 7. Order Block Strength (订单块强度):
        # 大涨之前最后一根阴线的成交量 (用 volume_delta proxy)
        # 如果上一分钟是阴线且本分钟是阳线, 用上一分钟的成交量作为订单块强度
        pl.when(
            (pl.col("period_last_price").shift(1).over("万得代码") < pl.col("period_first_price").shift(1).over("万得代码")) &
            (pl.col("period_last_price") > pl.col("period_first_price"))
        ).then(pl.col("total_trade_vol").shift(1).over("万得代码"))
        .otherwise(pl.lit(0.0)).alias("smc_order_block_strength"),
        
        # 8. Wyckoff Accumulation Proxy (威科夫吸筹度):
        # 量能变异系数 / 价格波幅 → 低值 = 量缩价稳 = 吸筹阶段
        (pl.col("total_trade_vol").rolling_std(window_size=10).over("万得代码").fill_null(0.0) /
         (pl.col("total_trade_vol").rolling_mean(window_size=10).over("万得代码").fill_null(1e-9) + 1e-9) /
         (((pl.col("period_high_price").rolling_max(window_size=10).over("万得代码") -
            pl.col("period_low_price").rolling_min(window_size=10).over("万得代码")) /
           (pl.col("period_last_price") + 1e-9)) + 1e-6)).alias("smc_wyckoff_accum")
           
    )
    
    # === [V11] 跨维度截面因子聚合 (Cross-sectional Rank) ===
    # 按照每分钟时间戳 `period_start` 对所有股票的截面状态进行特征提取
    df_all_tick = df_all_tick.with_columns(
        # 计算该股票近10分钟的累计交易额
        pl.col("total_trade_amt").rolling_sum(window_size=10).over("万得代码").alias("rolling_amt_10m"),
        # 计算该股票当分钟对开盘价的相对涨幅
        ((pl.col("period_last_price") - pl.col("day_open_price")) / (pl.col("day_open_price") + 1e-9)).alias("intraday_return")
    ).with_columns(
        # 3. 吸血效应排名 (Relative Volume Rank 10m): 在全市场的10分钟滚动成交排名百分比
        (pl.col("rolling_amt_10m").rank(descending=False).over("period_start") / 
         (pl.col("rolling_amt_10m").count().over("period_start") + 1e-9)).alias("relative_volume_rank_10m"),
         
        # 4. 截面抗跌强度 (Intraday Strength Z-score): 
        # (个股涨幅 - 截面均涨幅) / 截面标准差
        ((pl.col("intraday_return") - pl.col("intraday_return").mean().over("period_start")) / 
         (pl.col("intraday_return").std().over("period_start") + 1e-6)).alias("intraday_strength_zscore")
    ).drop([
        "volume_delta", "cvd", "cum_vol", "day_open_price", "total_trade_amt", 
        "period_last_price", "period_first_price", "period_high_price", "period_low_price",
        "down_vol_sum", "up_vol_sum", "avg_active_buy_size", "rolling_amt_10m", "intraday_return"
    ])
    
    return df_all_tick
