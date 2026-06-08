import os
import multiprocessing

# === 反递归哨兵 ===
# Windows spawn 机制会在子进程中重新导入此模块；
# 如果检测到当前是子进程，则跳过全部重量级初始化，
# 从而彻底阻断 spawn 递归导致的并行爆炸。
_is_main_process = (multiprocessing.parent_process() is None)

if _is_main_process:
    os.environ["POLARS_MAX_THREADS"] = "2"
    os.environ["POLARS_TEMP_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "_polars_temp")
    os.makedirs(os.environ["POLARS_TEMP_DIR"], exist_ok=True)

import polars as pl
import time

from engine import compute_and_aggregate_ashare
from tick_factors import compute_tick_factors_for_date
from ml_pipeline import generate_rolling_features, V3_FEATURES

SNAPSHOT_BASE = "E:/AGUDATA/l2_snapshot"
TICK_BASE = "E:/AGUDATA/l2_tick"
# 恢复至当前项目的 models 文件夹
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
FEATURE_OUTPUT = os.path.join(OUTPUT_DIR, "training_data.parquet")

def get_all_dates(base_dir: str, start_date: str = "20240102") -> list:
    """获取所有可用交易日，优先从原始数据源读取，降级从 daily_cache 推导"""
    dates: list[str] = []
    if os.path.isdir(base_dir):
        for d in sorted(os.listdir(base_dir)):
            if d.startswith("date="):
                date_str = d.replace("date=", "")
                if date_str >= start_date:
                    dates.append(date_str)
    # 降级方案: 原始数据源不可用时从 daily_cache 推导
    if not dates:
        cache_dir = os.path.join(OUTPUT_DIR, "daily_cache")
        if os.path.isdir(cache_dir):
            for f in sorted(os.listdir(cache_dir)):
                if f.endswith(".parquet") and len(f) == len("YYYYMMDD.parquet"):
                    date_str = f.replace(".parquet", "")
                    if date_str >= start_date:
                        dates.append(date_str)
            if dates:
                print(f"[get_all_dates] 原始数据源不可用，从 daily_cache 推导出 {len(dates)} 个交易日")
    return dates

def _process_and_save_single_file(f: str, idx: int, temp_buffer_dir: str, needed_cols: list, price_cols: list, vol_cols: list, date_cols: list, time_step: int) -> bool:
    import polars as pl
    import os
    import gc
    from engine import compute_stock_level_factors
    
    if not os.path.isfile(f):
        print(f"    [Skip] 文件不存在: {f}")
        return False
    
    try:
        # [V14 Memory Fix] 加强垃圾回收并使用 pyarrow 后端加载，避免 Polars/Jemalloc 的 mmap arena 在反复释放时崩溃
        df_file = pl.read_parquet(f, use_pyarrow=True)
        available = [c for c in needed_cols if c in df_file.columns]
        df_file = df_file.select(available)
        
        num_to_cast = [c for c in (price_cols + vol_cols + ['当日累计成交量', '品种总数']) if c in available]
        if num_to_cast:
            df_file = df_file.with_columns([pl.col(c).cast(pl.Float32, strict=False) for c in num_to_cast])
        
        date_to_cast = [c for c in date_cols if c in available]
        if date_to_cast:
            for c in date_to_cast:
                df_file = df_file.with_columns(pl.col(c).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False))
        
        if "时间" in available and "自然日" in available:
            df_file = df_file.filter(pl.col("时间").is_not_null() & pl.col("自然日").is_not_null())
            
        # [V18 Hotfix] 最底层 Zero-Price 物理隔离，防止坏点毒化后续高阶聚合特征
        if "成交价" in available and "前收盘" in available:
            df_file = df_file.filter((pl.col("成交价") > 0) & (pl.col("前收盘") > 0))
            
        rename_map = {}
        for old_name in ["前收盘", "前收盘价", "昨收"]:
            if old_name in df_file.columns:
                rename_map[old_name] = "pre_close"
        if "成交价" in df_file.columns:
            rename_map["成交价"] = "last_price"
            
        if rename_map:
            df_file = df_file.rename(rename_map)
        
        df_file_agg = compute_stock_level_factors(df_file, time_step)
        
        if df_file_agg.height > 0:
            out_path = os.path.join(temp_buffer_dir, f"part_{idx}.parquet")
            df_file_agg.write_parquet(out_path)
            
        del df_file, df_file_agg
        gc.collect()
        return True
    except Exception as e:
        print(f"    [Error] 文件处理失败 {f}: {e}")
        return False


def process_single_day(date_str: str, time_step: int = 60, return_last_only: bool = False) -> pl.DataFrame:
    """处理单日数据：Snapshot 因子 + Tick 因子融合 (磁盘缓冲聚合 V10 版)"""
    import gc
    import shutil
    from engine import compute_stock_level_factors, compute_market_level_factors
    
    snap_dir = os.path.join(SNAPSHOT_BASE, f"date={date_str}")
    if not os.path.exists(snap_dir):
        return pl.DataFrame()
    parquet_files = [
        os.path.join(snap_dir, f) for f in os.listdir(snap_dir)
        if f.endswith(".parquet") and os.path.isfile(os.path.join(snap_dir, f))
    ]
    
    if not parquet_files:
        return pl.DataFrame()
    
    import uuid
    # 临时缓冲目录
    temp_buffer_dir = os.path.join(OUTPUT_DIR, f"temp_buffer_{date_str}_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    if os.path.exists(temp_buffer_dir):
        shutil.rmtree(temp_buffer_dir)
    os.makedirs(temp_buffer_dir, exist_ok=True)
    
    price_cols = [f'申买价{i}' for i in range(1, 11)] + [f'申卖价{i}' for i in range(1, 11)] + \
                 ['成交价', '前收盘', '昨收', '叫买总量', '叫卖总量']
    vol_cols = [f'申买量{i}' for i in range(1, 11)] + [f'申卖量{i}' for i in range(1, 11)]
    date_cols = ['自然日', '时间'] 
    needed_cols = price_cols + vol_cols + date_cols + ['万得代码', '当日累计成交量', '品种总数']
    
    print(f"    [Disk-Buffered] 正在逐个缓存 {len(parquet_files)} 个文件的聚合结果...")
    
    # [V13 OOM Fix] 串行处理，避免多线程同时驻留多份 Snapshot 数据导致内存峰值
    results = []
    for idx, f in enumerate(parquet_files):
        ok = _process_and_save_single_file(f, idx, temp_buffer_dir, needed_cols, price_cols, vol_cols, date_cols, time_step)
        results.append(ok)
        if (idx + 1) % 10 == 0:  # 改为更频繁的 gc
            gc.collect()
            print(f"    [Snap] 已处理 {idx+1}/{len(parquet_files)} 个文件...")
    count = sum(1 for r in results if r)
    
    if count == 0:
        shutil.rmtree(temp_buffer_dir)
        return pl.DataFrame()
        
    # 4. 直接加载并合并聚合后的小文件 (文件极小，pl.concat 最稳健)
    # 🌟 优化: 让 Polars 内部处理多文件读取，比 Python 列表推导式更节省内存
    df_day_agg = pl.read_parquet(os.path.join(temp_buffer_dir, "*.parquet"), low_memory=True)
    
    # 🌟 核心运行: 生成 Rolling 特征与市场维度特征 🌟
    from ml_pipeline import generate_rolling_features
    df_day_agg = df_day_agg.sort(["万得代码", "period_start"])
    df_day_agg = generate_rolling_features(df_day_agg)
    
    # 计算全市场/宏观维度因子
    df_snap = compute_market_level_factors(df_day_agg)
    df_snap = generate_rolling_features(df_snap)
    
    # 🌟 极简方案: 计算完 snap 后，立即释放巨大的全天明细数据
    del df_day_agg
    gc.collect()
    
    # === 增强: 修复 Windows 文件锁定问题 (WinError 32) ===
    # 在 df_day_agg 被删除后再尝试删除临时目录，此时 mmap 应该已释放
    try:
        if os.path.exists(temp_buffer_dir):
            shutil.rmtree(temp_buffer_dir)
    except:
        pass
    
    # Tick 因子计算与融合
    try:
        df_tick = compute_tick_factors_for_date(date_str, TICK_BASE)
        if df_tick.height > 0:
            tick_cols = [c for c in df_tick.columns if c not in ("万得代码", "period_start")]
            df_snap = df_snap.join(df_tick, on=["万得代码", "period_start"], how="left")
            df_snap = df_snap.with_columns([pl.col(c).fill_null(0.0) for c in tick_cols])
            
            # [V7] 提取 Cross-Domain 跨域组合因子
            # 1. Trade-Order Divergence: (Tick 量流差) - (Snapshot 挂单流差)
            if "active_buy_total" in df_snap.columns and "active_sell_total" in df_snap.columns and "ofi_sum" in df_snap.columns:
                 df_snap = df_snap.with_columns(
                     ((pl.col("active_buy_total") - pl.col("active_sell_total")) - pl.col("ofi_sum")).alias("cross_trade_order_divergence")
                 )
                 
            # 2. Book Depletion Rate Proxy: (Tick 成交量) * (Snapshot 盘口斜率)
            if "total_trade_vol" in df_snap.columns and "book_slope_mean" in df_snap.columns:
                 df_snap = df_snap.with_columns(
                     (pl.col("total_trade_vol") * pl.col("book_slope_mean")).alias("cross_book_depletion_rate")
                 )
                 
            # 3. Hidden Accumulation: (冰山挂单撤补) * (Tick CVD 背离)
            if "iceberg_replenish_proxy" in df_snap.columns and "cvd_divergence" in df_snap.columns:
                 df_snap = df_snap.with_columns(
                     (pl.col("iceberg_replenish_proxy") * pl.col("cvd_divergence")).alias("cross_hidden_accumulation")
                 )
            
            # [V19] Cross-Domain Aggressiveness: Tick 单笔扫单总金额 / 盘口首尾厚度
            if "max_single_trade_amt" in df_snap.columns and "close_bv1" in df_snap.columns and "close_bp1" in df_snap.columns and "close_av1" in df_snap.columns and "close_ap1" in df_snap.columns:
                 df_snap = df_snap.with_columns(
                     (pl.col("max_single_trade_amt") / ((pl.col("close_bv1") * pl.col("close_bp1")) + (pl.col("close_av1") * pl.col("close_ap1")) + 1e-9)).alias("cross_aggressiveness_ratio")
                 )
                 
            print(f"    [Merge] {date_str} Tick 因子融合且 V7 Cross-Domain 计算成功")
        else:
            print(f"    [Warning] {date_str} Tick 因子计算结果为空，将使用零值占位")
    except Exception as e:
        print(f"    [Merge Error] {date_str} Tick 融合异常: {e}")
    
    # 标记日期
    df_snap = df_snap.with_columns(pl.lit(date_str).alias("trade_date"))
    
    # === 🌟 核心补丁: 强制补齐 V3 因子集，防止模型训练环节崩溃 🌟 ===
    missing_cols = [c for c in V3_FEATURES if c not in df_snap.columns]
    if missing_cols:
        print(f"    [Schema Fix] 正在补齐缺失字段: {missing_cols}")
        df_snap = df_snap.with_columns([pl.lit(0.0).alias(c) for c in missing_cols])
    
    # === 极简内存模式: 如果只需要截面，直接在这里处理并释放全天数据 ===
    if return_last_only:
        # 在只保留最后一行前，计算并保留全天的开高低收，供标签生成使用
        df_last = df_snap.sort("period_start").group_by("万得代码").last()
        if "last_price" in df_snap.columns:
            df_valid = df_snap.filter(pl.col("last_price") > 1e-3)
            df_prices = df_valid.group_by("万得代码").agg([
                pl.col("last_price").first().alias("day_open"),
                pl.col("last_price").max().alias("day_high"),
                pl.col("last_price").min().alias("day_low"),
                pl.col("last_price").last().alias("day_close"),
            ])
            if "day_open" in df_last.columns:
                df_last = df_last.drop(["day_open", "day_high", "day_low", "day_close"])
            df_last = df_last.join(df_prices, on="万得代码", how="left")
            del df_valid, df_prices
        del df_snap
        return df_last
    
    return df_snap

def generate_real_labels(df_input: pl.DataFrame) -> pl.DataFrame:
    """
    真实标签生成器 V2：
    - future_max_return: 利用次日截面中每只股票的最高 last_price 估算
    - future_max_drawdown: 利用次日截面中每只股票的最低 last_price 估算
    """
    print("[Label] 正在生成真实标签 (含真实回撤)...")
    
    # 判断是否需要提取截面
    if "trade_date" in df_input.columns and df_input.height < 5000 * 300:
        df_daily = df_input
    else:
        group_cols = ["万得代码"]
        if "trade_date" in df_input.columns:
            group_cols.append("trade_date")
            
        df_daily = (
            df_input
            .sort("period_start")
            .group_by(group_cols)
            .last()
        )
    
    # 如果已经有了优化后预存的 day_high，直接不用计算了！
    if "day_high" in df_input.columns and "day_low" in df_input.columns:
        df_hilo = df_daily.select(["万得代码", "trade_date", "day_open", "day_high", "day_low", "day_close"])
    elif "trade_date" in df_input.columns and "last_price" in df_input.columns:
        df_hilo = (
            df_input.filter(pl.col("last_price") > 0)
            .group_by(["万得代码", "trade_date"])
            .agg([
                pl.col("last_price").first().alias("day_open"),
                pl.col("last_price").max().alias("day_high"),
                pl.col("last_price").min().alias("day_low"),
                pl.col("last_price").last().alias("day_close"),
            ])
        )
    else:
        df_hilo = None

    dates = sorted(df_daily.select("trade_date").unique().to_series().to_list())
    
    results = []
    for i, dt in enumerate(dates):
        df_today = df_daily.filter(pl.col("trade_date") == dt)
        
        if i + 2 < len(dates):
            next_dt = dates[i + 1]
            next_next_dt = dates[i + 2]
            
            if df_hilo is not None:
                # T日（建仓日）用于取买入开盘价及承受锁仓盘中回撤
                df_next_hilo = df_hilo.filter(pl.col("trade_date") == next_dt).select([
                    pl.col("万得代码"),
                    pl.col("day_open").alias("next_day_open"),
                    pl.col("day_low").alias("next_day_low"),
                    pl.col("day_close").alias("next_day_close"),
                ])
                # T+1日（平仓日）用于取卖出高点及收盘
                df_next_next_hilo = df_hilo.filter(pl.col("trade_date") == next_next_dt).select([
                    pl.col("万得代码"),
                    pl.col("day_open").alias("next_next_day_open"),
                    pl.col("day_high").alias("next_next_day_high"),
                    pl.col("day_low").alias("next_next_day_low"),
                    pl.col("day_close").alias("next_next_day_close"),
                ])
                
                df_labeled = df_today.join(df_next_hilo, on="万得代码", how="left")
                df_labeled = df_labeled.join(df_next_next_hilo, on="万得代码", how="left")
                df_labeled = df_labeled.with_columns(
                    pl.when(pl.col("next_day_open") > 1e-3)
                      .then((pl.col("next_next_day_high") / pl.col("next_day_open")) - 1)
                      .otherwise(0.0).alias("future_max_return"),
                    pl.when(pl.col("next_day_open") > 1e-3)
                      .then((pl.col("next_next_day_low") / pl.col("next_day_open")) - 1) # [V12] 修复: 严格剥离 T+1 的盘中回撤，只使用 T+2 的真正可卖出低点
                      .otherwise(0.0).alias("future_max_drawdown"),
                    pl.when(pl.col("next_day_open") > 1e-3)
                      .then((pl.col("next_next_day_close") / pl.col("next_day_open")) - 1)
                      .otherwise(0.0).alias("future_close_return"),
                    pl.when(pl.col("next_day_open") > 1e-3)
                      .then((pl.col("next_day_close") / pl.col("next_day_open")) - 1)
                      .otherwise(0.0).alias("t_day_close_return"),
                    pl.when(pl.col("next_day_open") > 1e-3)
                      .then((pl.col("next_next_day_open") / pl.col("next_day_open")) - 1)
                      .otherwise(0.0).alias("next_next_day_open_return"),
                )
            else:
                # 降级方案: 只有收盘价
                df_next_prices = df_daily.filter(pl.col("trade_date") == next_next_dt).select([
                    pl.col("万得代码"),
                    pl.col("last_price").alias("next_next_day_close"),
                ])
                df_labeled = df_today.join(df_next_prices, on="万得代码", how="left")
                df_labeled = df_labeled.with_columns(
                    pl.when(pl.col("last_price") > 1e-3)
                      .then((pl.col("next_next_day_close") / pl.col("last_price")) - 1)
                      .otherwise(0.0).alias("future_max_return"),
                    pl.when(pl.col("last_price") > 1e-3)
                      .then((pl.col("next_next_day_close") / pl.col("last_price")) - 1)
                      .otherwise(0.0).alias("future_max_drawdown"),
                    pl.when(pl.col("last_price") > 1e-3)
                      .then((pl.col("next_next_day_close") / pl.col("last_price")) - 1)
                      .otherwise(0.0).alias("future_close_return"),
                    pl.lit(0.0).alias("t_day_close_return"),
                    pl.lit(0.0).alias("next_next_day_open_return"),
                )
        else:
            df_labeled = df_today.with_columns(
                pl.lit(0.0).alias("future_max_return"),
                pl.lit(0.0).alias("future_max_drawdown"),
                pl.lit(0.0).alias("future_close_return"),
                pl.lit(0.0).alias("t_day_close_return"),
                pl.lit(0.0).alias("next_next_day_open_return"),
            )
        
        results.append(df_labeled)
    
    df_final = pl.concat(results, how="diagonal_relaxed")
    
    # [V9] 使用 V9 版本的动态自适应标签，代替原先写死的 10% / -3%
    from ml_pipeline import generate_labels_v3
    df_final = generate_labels_v3(df_final)
    
    return df_final

def run_batch(max_days: int = 0):
    """
    全量批处理主函数
    max_days: 0 表示处理全部，>0 表示只处理前 N 天 (用于测试)
    """
    all_dates = get_all_dates(SNAPSHOT_BASE)
    if max_days > 0:
        all_dates = all_dates[:max_days]
    
    print(f"[Batch] 共检测到 {len(all_dates)} 个交易日，开始逐日处理...")
    
    all_dfs = []
    for idx, dt in enumerate(all_dates):
        t0 = time.time()
        print(f"\n--- [{idx+1}/{len(all_dates)}] 处理日期: {dt} ---")
        try:
            df_day = process_single_day(dt)
            if df_day.height > 0:
                all_dfs.append(df_day)
                print(f"    完成! 行数={df_day.height}, 耗时={time.time()-t0:.1f}s")
        except Exception as e:
            print(f"    [ERROR] 处理 {dt} 失败: {e}")
    
    if not all_dfs:
        print("[Batch] 没有成功处理任何日期!")
        return
    
    print("\n[Batch] 合并所有日期数据...")
    df_all = pl.concat(all_dfs, how="diagonal_relaxed")
    
    print("[Batch] 生成真实标签...")
    df_labeled = generate_real_labels(df_all)
    
    df_labeled.write_parquet(FEATURE_OUTPUT)
    print(f"\n[Batch] 全量训练数据已保存至: {FEATURE_OUTPUT}")
    print(f"    总行数: {df_labeled.height}, 总列数: {len(df_labeled.columns)}")
    
def run_oos_backtest(train_days: int = 200):
    """
    OOS (Out-of-Sample) 验证逻辑：
    - 针对大规模数据进行了极致内存优化：逐日提取截面并释放原始数据
    """
    import gc
    import traceback
    from ml_pipeline import V3_FEATURES, train_ensemble_v3, predict_ensemble_top3
    ERROR_LOG = os.path.join(OUTPUT_DIR, "oos_crash_error.log")
    
    all_dates = get_all_dates(SNAPSHOT_BASE)
    if len(all_dates) <= train_days:
        print(f"[OOS] 数据不足 (总={len(all_dates)}, 训练={train_days})")
        return

    train_dates = all_dates[:train_days]
    test_dates = all_dates[train_days:]
    
    print(f"[OOS] 训练阶段: {len(train_dates)} 天")
    
    # 1. 训练阶段: 逐日提取特征截面 (只留最后一行，极大节省内存)
    print("--- [Step 1/2] 正在构建训练集 (截面提取)... ---")
    train_dfs = []
    for idx, dt in enumerate(train_dates):
        try:
            df_day = process_single_day(dt)
            if df_day.height > 0:
                # 只提取尾盘截面用于模型学习
                df_last = df_day.sort("period_start").group_by("万得代码").last()
                if df_last.height > 0:
                    train_dfs.append(df_last)
                    print(f"    [{idx+1}/{len(train_dates)}] {dt} 截面提取完成 (行数={df_last.height})")
                else:
                    print(f"    [{idx+1}/{len(train_dates)}] {dt} 截面为空，跳过")
                del df_day, df_last
            else:
                print(f"    [{idx+1}/{len(train_dates)}] {dt} 数据读取为空，跳过")
            gc.collect()
        except Exception as e:
            error_msg = f"\n[CRITICAL ERROR] Day {idx+1} ({dt}) 发生异常:\n{traceback.format_exc()}\n"
            print(error_msg)
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(error_msg)
            # 是否继续还是停止？如果是致命错误（如内存不足）建议停止
            if "MemoryError" in str(e) or "allocation" in str(e).lower():
                return

    if not train_dfs:
        print("[OOS] 训练集为空，无法继续。")
        return

    # 合并训练截面
    print("[OOS] 正在合并训练集...")
    df_train_daily = pl.concat(train_dfs, how="diagonal_relaxed")
    del train_dfs
    gc.collect()
    
    # 生成标签
    df_train_labeled = generate_real_labels(df_train_daily)
    
    # === Alpha Mining: 自动发现新因子 ===
    print("\n[OOS] 启动 Alpha Mining Engine (带 Pre-screening 内存防护)...")
    from factor_generator import run_alpha_mining, generate_cross_factors, load_discovered_factors, screen_by_ic
    
    discovered = run_alpha_mining(df_train_labeled, base_features=V3_FEATURES, label_col="label")
    all_features = V3_FEATURES + discovered
    print(f"[OOS] 最终特征集: {len(V3_FEATURES)} 基础 + {len(discovered)} 挖掘 = {len(all_features)} 总计")
    
    # 为训练数据生成交叉因子列 (如果 Alpha Mining 发现了新因子)
    if discovered:
        df_train_labeled, _ = generate_cross_factors(df_train_labeled, V3_FEATURES, target_names=discovered)
    
    # 训练模型
    print("[OOS] 开始训练模型集...")
    train_ensemble_v3(df_train_labeled, features=all_features, train_date=train_dates[-1])
    
    # 清理训练内存
    del df_train_daily, df_train_labeled
    gc.collect()
    
    # 2. 测试阶段 (OOS): 逐日预测并回测 (流式处理 + 累计净值追踪)
    print("\n--- [Step 2/2] 开始测试阶段 (OOS 滚动验证)... ---")
    from backtest_simulation import StatefulPortfolioManager
    
    pm = StatefulPortfolioManager(initial_capital=1000000.0)
    cumulative_capital = pm.initial_capital
    equity_curve = []  # (date, capital)
    daily_returns = []
    previous_cap = pm.initial_capital
    
    for dt in test_dates:
        try:
            print(f"\n[OOS Test] 日期: {dt}")
            # Use return_last_only=True so it computes daily Ohlc correctly
            df_test_cs = process_single_day(dt, return_last_only=True)
            if df_test_cs.height <= 0:
                continue
            
            # 1. 喂给状态机今日行情数据，执行挂单，并盘中离场
            pm.process_daily_market(df_test_cs, dt)
                
            # 注入交叉因子 (与训练阶段保持一致)
            if discovered:
                df_test_cs, _ = generate_cross_factors(df_test_cs, V3_FEATURES)
            
            df_top3 = predict_ensemble_top3(df_test_cs, features=all_features)
            
            # [V9] 提取当天的市场大盘环境 (Market Regime)
            current_regime = -1
            if "market_regime" in df_test_cs.columns:
                try:
                    regime_counts = df_test_cs.select("market_regime").to_series().value_counts()
                    current_regime = regime_counts.sort("count", descending=True)["market_regime"][0]
                except Exception:
                    pass
            
            # 2. 状态机分发次日的新买单
            pm.generate_target_orders(df_top3, current_market_regime=current_regime)
            
            # 追踪累计净值
            cumulative_capital = pm.get_equity(df_test_cs)
            day_return = (cumulative_capital - previous_cap) / previous_cap if previous_cap > 0 else 0.0
            daily_returns.append(day_return)
            equity_curve.append({"date": dt, "capital": cumulative_capital, "daily_return": day_return})
            previous_cap = cumulative_capital
            
            print(f"    [Equity] 累计净值: {cumulative_capital:.2f} ￥ | {pm.get_summary()}")
            
            # 显式清理
            del df_test_cs
            gc.collect()
        except Exception as e:
            error_msg = f"\n[CRITICAL ERROR] Test Day ({dt}) 发生异常:\n{traceback.format_exc()}\n"
            print(error_msg)
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(error_msg)
            continue

    # ========== OOS 完成: 输出 KPI 汇总 ==========
    print("\n" + "=" * 70)
    print("【OOS 验证最终报告】")
    print("=" * 70)
    
    if daily_returns:
        import numpy as np
        returns_arr = np.array(daily_returns)
        total_return = (cumulative_capital / 1000000.0) - 1.0
        n_days = len(daily_returns)
        ann_return = (1.0 + total_return) ** (252 / max(n_days, 1)) - 1.0
        
        # 最大回撤
        peak = 1000000.0
        max_dd = 0.0
        cap = 1000000.0
        for r in daily_returns:
            cap *= (1.0 + r)
            if cap > peak:
                peak = cap
            dd = (cap - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        # 夏普比率 (无风险利率 ≈ 2%)
        daily_std = np.std(returns_arr) if len(returns_arr) > 1 else 1e-9
        sharpe = (np.mean(returns_arr) - 0.02 / 252) / (daily_std + 1e-9) * np.sqrt(252)
        
        win_rate = np.sum(returns_arr > 0) / len(returns_arr) * 100
        
        print(f"  测试天数:     {n_days}")
        print(f"  累计净值:     {cumulative_capital:.2f} ￥")
        print(f"  总收益率:     {total_return * 100:.2f}%")
        print(f"  年化收益率:   {ann_return * 100:.2f}%")
        print(f"  最大回撤:     {max_dd * 100:.2f}%")
        print(f"  夏普比率:     {sharpe:.3f}")
        print(f"  胜率:         {win_rate:.1f}%")
        print(f"  Calmar 比率:  {ann_return / abs(max_dd + 1e-9):.3f}")
        
        # 保存净值曲线
        equity_path = os.path.join(OUTPUT_DIR, "oos_equity_curve.csv")
        df_equity = pl.DataFrame(equity_curve)
        df_equity.write_csv(equity_path)
        print(f"\n  净值曲线已保存: {equity_path}")
    else:
        print("  没有有效的测试日数据。")


def _compute_and_save_cache_worker(date_str: str, cache_path: str):
    import polars as pl
    import os
    import gc
    df_day = process_single_day(date_str, return_last_only=True)
    if df_day.height > 0:
        df_day.write_parquet(cache_path)
    del df_day
    gc.collect()


def run_walkforward_backtest(train_window: int = 200, retrain_every: int = 20, start_date: str = None):
    """
    Walk-Forward 滚动重训练回测引擎:
    - 每 retrain_every 个测试日，用最近 train_window 天重新训练模型
    - 训练时同步执行 Alpha Mining 发现新因子
    - 逐日预测 + 回测 + 追踪累计净值
    """
    import gc
    import traceback
    import numpy as np
    ERROR_LOG = os.path.join(OUTPUT_DIR, "wf_crash_error.log")

    all_dates = get_all_dates(SNAPSHOT_BASE)
    
    # [V11] 支持指定起始日期截断
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
        print(f"[WF] 已切分起始日期: >= {start_date} (剩余 {len(all_dates)} 天)")
        
    if len(all_dates) <= train_window:
        print(f"[WF] 数据不足 (总={len(all_dates)}, 训练窗口={train_window})")
        return

    print("=" * 70)
    print(f"【Walk-Forward 滚动重训练回测引擎】")
    print(f"  训练窗口: {train_window} 天 | 重训频率: 每 {retrain_every} 天")
    print(f"  总交易日: {len(all_dates)} | 测试期: Day {train_window+1} ~ Day {len(all_dates)}")
    print("=" * 70)

    from ml_pipeline import V3_FEATURES, train_ensemble_v3, predict_ensemble_top3
    from factor_generator import run_alpha_mining, generate_cross_factors
    from universe_filter import filter_universe

    def get_cross_section(date_str: str) -> pl.DataFrame:
        """获取某日的截面数据 (磁盘缓存优先版)"""
        cache_dir = os.path.join(OUTPUT_DIR, "daily_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{date_str}.parquet")
        
        if os.path.exists(cache_path):
            try:
                df_cache = pl.read_parquet(cache_path)
                if df_cache.height > 0:
                    toxic_exprs = []
                    if "vpin" in df_cache.columns and "intraday_momentum" in df_cache.columns and "toxic_breakout_trap" not in df_cache.columns:
                        toxic_exprs.append((pl.col("vpin") / (pl.col("intraday_momentum").abs() + 1e-9)).alias("toxic_breakout_trap"))
                    if "limit_up_magnet_effect" in df_cache.columns and "smc_fvg_bearish" in df_cache.columns and "false_magnet_drop" not in df_cache.columns:
                        toxic_exprs.append((pl.col("limit_up_magnet_effect") * pl.col("smc_fvg_bearish")).alias("false_magnet_drop"))
                    if toxic_exprs:
                        df_cache = df_cache.with_columns(toxic_exprs)
                        
                    return df_cache
            except Exception as e:
                print(f"    [Cache Error] 读取缓存失败 {date_str}: {e}，将重新计算...")
        # 缓存不存在，使用独立进程进行计算，彻底杜绝 Jemalloc arena 历史遗留胀气
        import multiprocessing
        p = multiprocessing.Process(target=_compute_and_save_cache_worker, args=(date_str, cache_path))
        p.start()
        p.join()
        
        if os.path.exists(cache_path):
            try:
                # 重试加载刚刚通过子进程生成的 Parquet
                df_cache = pl.read_parquet(cache_path)
                if df_cache.height > 0:
                    toxic_exprs = []
                    if "vpin" in df_cache.columns and "intraday_momentum" in df_cache.columns and "toxic_breakout_trap" not in df_cache.columns:
                        toxic_exprs.append((pl.col("vpin") / (pl.col("intraday_momentum").abs() + 1e-9)).alias("toxic_breakout_trap"))
                    if "limit_up_magnet_effect" in df_cache.columns and "smc_fvg_bearish" in df_cache.columns and "false_magnet_drop" not in df_cache.columns:
                        toxic_exprs.append((pl.col("limit_up_magnet_effect") * pl.col("smc_fvg_bearish")).alias("false_magnet_drop"))
                    if toxic_exprs:
                        df_cache = df_cache.with_columns(toxic_exprs)
                    return df_cache
            except Exception as e:
                print(f"    [Cache Error] 读取刚生成的缓存失败 {date_str}: {e}")
        return pl.DataFrame()

    # 追踪变量
    from backtest_simulation import StatefulPortfolioManager
    
    cumulative_capital = 1000000.0
    equity_curve: list[dict] = []
    daily_returns: list[float] = []
    market_breadth_history: list[float] = []
    discovered: list[str] = []
    all_features = V3_FEATURES.copy()

    test_start = train_window
    resume_dt = None
    
    # [Resume Logic] 解析以往 wf_pnl.log 恢复断点
    pnl_log_path = os.path.join(OUTPUT_DIR, "wf_pnl.log")
    if os.path.exists(pnl_log_path):
        try:
            with open(pnl_log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if lines:
                print(f"[WF] 发现断点记录 {pnl_log_path}，包含 {len(lines)} 天记录，尝试恢复进度...")
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) >= 4:
                        dt = parts[0]
                        cap = float(parts[1])
                        d_ret = float(parts[2])
                        equity_curve.append({"date": dt, "capital": cap, "daily_return": d_ret})
                        daily_returns.append(d_ret)
                
                last_record = lines[-1].strip().split(",")
                resume_dt = last_record[0]
                cumulative_capital = float(last_record[1])
                print(f"  -> 恢复至 {resume_dt} | 资金: {cumulative_capital:.2f}")
                
                if resume_dt in all_dates:
                    test_start = all_dates.index(resume_dt) + 1
                    print(f"  -> test_start 调整为 {test_start}")
                
                # 恢复 discovered factors 列表，防止特征数量为0的 KeyError 或者缺失列
                import json
                try:
                    with open(os.path.join(OUTPUT_DIR, "discovered_factors.json"), "r") as f:
                        saved_meta = json.load(f)
                    factor_deps = saved_meta.get("factor_dependencies", {})
                    if factor_deps:
                        discovered = list(factor_deps.keys())
                        all_features = V3_FEATURES + discovered
                        print(f"  -> 恢复 discovered factors 列表: {len(discovered)} 个")
                except Exception:
                    pass
        except Exception as e:
            print(f"[WF] 恢复断点失败: {e}")

    # 以最新资金实例化资产管理器 (假定重启时没有持仓,符合日志记录平仓状态)
    pm = StatefulPortfolioManager(initial_capital=1000000.0)
    pm.cash = cumulative_capital
    previous_cap = cumulative_capital

    if test_start > train_window:
        rem_days = (test_start - train_window) % retrain_every
        days_since_retrain = retrain_every if rem_days == 0 else rem_days
    else:
        days_since_retrain = retrain_every  # 触发首次训练

    for test_idx in range(test_start, len(all_dates)):
        dt = all_dates[test_idx]

        # === 是否需要重新训练? ===
        if days_since_retrain >= retrain_every:
            train_start = max(0, test_idx - train_window)
            # [V8] Purge-Embargo: 砍掉训练窗口最后 2 天，防止标签泄漏
            # (标签用了 T+1 未来价格，最后 1 天偷看测试集，再加 1 天安全缓冲)
            train_end = max(train_start, test_idx - 2)
            train_dates_window = all_dates[train_start:train_end]

            print(f"\n{'='*60}")
            print(f"[WF Retrain] Window: Day {train_start+1}~{test_idx} ({len(train_dates_window)} days)")
            print(f"{'='*60}")

            # 构建训练集 (从缓存优先获取)
            train_dfs = []
            for idx, tdt in enumerate(train_dates_window):
                try:
                    df_cs = get_cross_section(tdt)
                    if df_cs.height > 0:
                        train_dfs.append(df_cs)
                    if (idx + 1) % 50 == 0:
                        print(f"    [Train] {idx+1}/{len(train_dates_window)} days loaded")
                except Exception as e:
                    print(f"    [Train] {tdt} failed: {e}")

            if not train_dfs:
                print("[WF] 训练集为空，跳过本轮。")
                days_since_retrain = 0
                continue

            df_train = pl.concat(train_dfs, how="diagonal_relaxed")
            df_train_labeled = generate_real_labels(df_train)
            del train_dfs
            # [FIX] Additional clean up
            del df_cs
            try:
                del df_day
            except:
                pass
            gc.collect()

            # [WF] Alpha Mining
            from factor_generator import screen_by_ic
            discovered = run_alpha_mining(
                df_train_labeled, base_features=V3_FEATURES,
                label_col="label", max_new_factors=15
            )
            all_features = V3_FEATURES + discovered

            # 注入交叉因子并训练
            if discovered:
                df_train_labeled, _ = generate_cross_factors(df_train_labeled, V3_FEATURES, target_names=discovered)

            train_ensemble_v3(df_train_labeled, features=all_features, train_date=train_dates_window[-1])

            del df_train, df_train_labeled
            gc.collect()
            days_since_retrain = 0

            # === 逐日预测 + 回测 ===
        try:
            df_test_cs = get_cross_section(dt)
            if df_test_cs.height <= 0:
                continue

            # ============= 核心新逻辑 =============
            # 1. 喂给状态机今日行情数据，执行昨天挂单，并在盘中评估所有的止盈止损
            pm.process_daily_market(df_test_cs, dt)

            # 选股宇宙过滤
            df_test_filtered = filter_universe(df_test_cs)
            if df_test_filtered.height <= 0:
                continue

            # 核心补丁：在 T+1 预测期，除了基础的因子，必须将训练期新发掘的 Alpha 因子全部在当天的截面上由原生表达式复刻出来。
            # 否则测试集的列将永远少于训练集，或者被强制填 0。
            if discovered:
                import json
                try:
                    with open(os.path.join("models", "discovered_factors.json"), "r") as f:
                        saved_meta = json.load(f)
                    
                    # [BugFix] 必须读取 factor_dependencies 子字典，而非遍历顶层 dict
                    factor_deps = saved_meta.get("factor_dependencies", {})
                    
                    expr_list = []
                    for f_name in discovered:
                        if f_name not in factor_deps:
                            continue
                        base_cols = factor_deps[f_name]
                        if len(base_cols) == 2:
                            c1, c2 = base_cols
                            if c1 not in df_test_filtered.columns or c2 not in df_test_filtered.columns:
                                continue
                            if "_div_" in f_name:
                                expr_list.append((pl.col(c1) / (pl.col(c2) + 1e-9)).alias(f_name))
                            elif "_sub_" in f_name:
                                expr_list.append((pl.col(c1) - pl.col(c2)).alias(f_name))
                            elif "_mul_" in f_name:
                                expr_list.append((pl.col(c1) * pl.col(c2)).alias(f_name))
                        elif len(base_cols) == 1:
                            c1 = base_cols[0]
                            if c1 not in df_test_filtered.columns:
                                continue
                            if f_name.endswith("_log1p_abs"):
                                expr_list.append(pl.col(c1).abs().log1p().cast(pl.Float32).alias(f_name))
                            elif f_name.endswith("_sign_sqrt"):
                                expr_list.append((pl.col(c1).sign() * pl.col(c1).abs().sqrt()).cast(pl.Float32).alias(f_name))
                            elif f_name.endswith("_rank"):
                                expr_list.append(pl.col(c1).rank().cast(pl.Float32).alias(f_name))
                    
                    if expr_list:
                        df_test_filtered = df_test_filtered.with_columns(expr_list)
                        print(f"  [Alpha Rebuild] 成功重建 {len(expr_list)} 个 discovered 因子")
                except Exception as ex:
                    print(f"  [Warning] 重建 Alpha 因子失败: {ex}")

            # [V9] 提取当天的市场大盘环境 (Market Regime)
            # 策略是: 看一下当天截面里众数的大盘状态 (因为市场情绪是针对全局的)
            current_regime = -1
            if "market_regime" in df_test_filtered.columns:
                try:
                    regime_counts = df_test_filtered.select("market_regime").to_series().value_counts()
                    current_regime = regime_counts.sort("count", descending=True)["market_regime"][0]
                except Exception:
                    pass

            # [V12] 绝对大盘原生状态指示器 (Market Breadth)
            current_breadth = 0.0
            if "last_price" in df_test_filtered.columns and "pre_close" in df_test_filtered.columns:
                try:
                    df_ret = df_test_filtered.filter(pl.col("pre_close") > 0.0)
                    if df_ret.height > 0:
                        df_ret = df_ret.with_columns(((pl.col("last_price") - pl.col("pre_close")) / pl.col("pre_close")).alias("daily_ret"))
                        current_breadth = df_ret.select(pl.col("daily_ret").median()).item()
                except Exception:
                    pass
            market_breadth_history.append(current_breadth)
            
            # 过去10天赚钱效应平滑，如果是深度阴跌则阻断
            breadth_ma_safe = True
            if len(market_breadth_history) >= 10:
                ma10 = sum(market_breadth_history[-10:]) / 10.0
                if ma10 < -0.003: # 连续10天中位涨跌幅不足 -0.3%
                    breadth_ma_safe = False

            # [V17.1] 终极释放: 完全移除夏普防线，解除由于空仓期 std=0 带来的无限自我恐慌封锁
            # 取消一切拦截特权，回归最原生态的宽度过滤

            df_top3 = predict_ensemble_top3(df_test_filtered, features=all_features)
            
            # 2. 让状态机处理新一轮的派单
            pm.generate_target_orders(df_top3, current_market_regime=current_regime, market_breadth_safe=breadth_ma_safe)

            # 3. 净值追踪
            cumulative_capital = pm.get_equity(df_test_cs)
            day_return = (cumulative_capital - previous_cap) / previous_cap if previous_cap > 0 else 0.0
            daily_returns.append(day_return)
            equity_curve.append({"date": dt, "capital": cumulative_capital, "daily_return": day_return})
            previous_cap = cumulative_capital

            summary_text = pm.get_summary()
            print(f"  [WF Day {test_idx+1}] {dt} | Equity: {cumulative_capital:.2f} | {summary_text}")

            # [V15 监控升级] 每日增量写入收益数据供实时查阅
            pnl_log_path = os.path.join(OUTPUT_DIR, "wf_pnl.log")
            try:
                with open(pnl_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{dt},{cumulative_capital:.2f},{day_return:.6f},{summary_text}\n")
            except Exception:
                pass

            try:
                del df_test_cs, df_test_filtered, df_top3
            except:
                pass
            gc.collect()

        except Exception as e:
            error_msg = f"\n[WF ERROR] {dt}:\n{traceback.format_exc()}\n"
            print(error_msg)
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(error_msg)

        days_since_retrain += 1

    # ========== KPI 汇总 ==========
    print("\n" + "=" * 70)
    print("【Walk-Forward 最终报告】")
    print("=" * 70)

    if daily_returns:
        returns_arr = np.array(daily_returns)
        total_return = (cumulative_capital / 1000000.0) - 1.0
        n_days = len(daily_returns)
        ann_return = (1.0 + total_return) ** (252 / max(n_days, 1)) - 1.0

        peak = 1000000.0
        max_dd = 0.0
        cap = 1000000.0
        for r in daily_returns:
            cap *= (1.0 + r)
            if cap > peak:
                peak = cap
            dd = (cap - peak) / peak
            if dd < max_dd:
                max_dd = dd

        daily_std = np.std(returns_arr) if len(returns_arr) > 1 else 1e-9
        sharpe = (np.mean(returns_arr) - 0.02 / 252) / (daily_std + 1e-9) * np.sqrt(252)
        win_rate = np.sum(returns_arr > 0) / len(returns_arr) * 100

        print(f"  测试天数:     {n_days}")
        print(f"  重训次数:     {n_days // retrain_every + 1}")
        print(f"  累计净值:     {cumulative_capital:.2f}")
        print(f"  总收益率:     {total_return * 100:.2f}%")
        print(f"  年化收益率:   {ann_return * 100:.2f}%")
        print(f"  最大回撤:     {max_dd * 100:.2f}%")
        print(f"  夏普比率:     {sharpe:.3f}")
        print(f"  胜率:         {win_rate:.1f}%")
        print(f"  Calmar:       {ann_return / abs(max_dd + 1e-9):.3f}")

        equity_path = os.path.join(OUTPUT_DIR, "wf_equity_curve.csv")
        df_equity = pl.DataFrame(equity_curve)
        df_equity.write_csv(equity_path)
        print(f"\n  净值曲线已保存: {equity_path}")
    else:
        print("  没有有效的测试日数据。")

    print("\n[WF] Walk-Forward 验证完成。")


if __name__ == "__main__":
    import sys
    import multiprocessing
    multiprocessing.freeze_support()

    mode = sys.argv[1] if len(sys.argv) > 1 else "batch"

    if mode == "oos":
        run_oos_backtest(train_days=200)
    elif mode == "wf":
        tw = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        re = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        sd = sys.argv[4] if len(sys.argv) > 4 else None
        run_walkforward_backtest(train_window=tw, retrain_every=re, start_date=sd)
    else:
        run_batch(max_days=0)
