import os
import polars as pl
import pandas as pd
from engine import compute_stock_level_factors

SNAPSHOT_BASE = "E:/AGUDATA/l2_snapshot"
TICK_BASE = "E:/AGUDATA/l2_tick"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
WATCHLIST_PATH = os.path.join(OUTPUT_DIR, "intraday_watchlist.csv")
INTRADAY_TRAIN_DATA = os.path.join(OUTPUT_DIR, "intraday_training_data.parquet")

def run_intraday_ignition_mvp(target_dates: list = None):
    print("=" * 60)
    print("[V21] Intraday Ignition Engine MVP")
    print("=" * 60)
    
    if not os.path.exists(WATCHLIST_PATH):
        print(f"Watchlist 不存在，请先运行 EOD 模型生成: {WATCHLIST_PATH}")
        return
        
    df_watch = pl.read_csv(WATCHLIST_PATH)
    available_dates = df_watch["date"].unique().to_list()
    
    if target_dates is None:
        target_dates = available_dates
        
    # [性能与 OOM 保护] 保证处理时间有序
    target_dates.sort()
    
    all_intraday_features = []
    
    import psutil
    
    for str_date in target_dates:
        str_date = str(str_date)
        symbols_today = df_watch.filter(pl.col("date") == int(str_date))["万得代码"].to_list()
        
        if not symbols_today:
            continue
            
        print(f"\n[Intraday Engine] 日期: {str_date} | 截获生态漏斗金股: {len(symbols_today)} 只")
        
        # 1. 内存友好的 Lazy Load (外存过滤)
        snap_path = os.path.join(SNAPSHOT_BASE, f"date={str_date}")
        if not os.path.exists(snap_path):
            print(f"  -> {str_date} 缺少 Snapshot 原始流，跳过。")
            continue
            
        try:
            parquet_files = [
                os.path.join(snap_path, f) for f in os.listdir(snap_path)
                if f.endswith(".parquet") and os.path.isfile(os.path.join(snap_path, f))
            ]
            
            df_list = []
            for f in parquet_files:
                try:
                    df_part = pl.read_parquet(f)
                    # Safe filter
                    if "万得代码" in df_part.columns:
                        df_part = df_part.filter(pl.col("万得代码").is_in(symbols_today))
                        if df_part.height > 0:
                            df_list.append(df_part)
                except Exception as e:
                    pass
                    
            if not df_list:
                print(f"  -> 未找到任何匹配的快照数据。")
                continue
                
            df_snap = pl.concat(df_list, how="diagonal_relaxed")
            
            mem_use = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            print(f"  -> L2 快照截取成功 (数据量: {df_snap.height} 行) | 当前内存占用: {mem_use:.1f} MB")
            
            # --- 若是在盘中点火期 (例如 9:30 - 10:30)，则仅截取早盘数据 ---
            # 真实实盘中，我们会流式传入，当前提取早盘数据作为模拟。
            # 这里先获取全天做打标，再模拟日内截断。
            
            # Float64->Int64 for trailing .0 
            if "自然日" in df_snap.columns:
                df_snap = df_snap.with_columns(
                    pl.col("自然日").cast(pl.Utf8).str.replace(r"\.0$", "").cast(pl.Int64)
                )
            if "时间" in df_snap.columns:
                df_snap = df_snap.with_columns(
                    pl.col("时间").cast(pl.Utf8).str.replace(r"\.0$", "").cast(pl.Int64)
                )
                
            # Cast price columns to float so engine division by 10000 doesn't crash on String
            price_cols = [f'申买价{i}' for i in range(1, 11)] + [f'申卖价{i}' for i in range(1, 11)] + ['pre_close', 'last_price', '当日累计成交量']
            vol_cols = [f'申买量{i}' for i in range(1, 11)] + [f'申卖量{i}' for i in range(1, 11)]
            for c in price_cols + vol_cols:
                if c in df_snap.columns:
                    df_snap = df_snap.with_columns(pl.col(c).cast(pl.Float64, strict=False))
                
            # 2. 借用底层引擎强大的微观提取器 (计算出分钟级状态序列)
            # engine.py 期待的必须是一个包含了 period_start 且按一定组别的 DF
            df_minute = compute_stock_level_factors(df_snap)
            
            mem_use = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            print(f"  -> 深度盘口动力学时序展开完成 (序列长度: {df_minute.height}) | 当前内存占用: {mem_use:.1f} MB")
            
            # 3. 日内打标模块 (Intraday Targeting)
            # 目标：这只股票在今天接下来的时间里，或者临近收盘，是否能封死涨停板 (赚钱效应终局)。
            # 由于我们使用的是分钟级序列，在每一分钟 t 上，我们预判 [t, close] 这段区间是否会触及涨停
            is_star_chi = df_minute['万得代码'].str.starts_with("688") | df_minute['万得代码'].str.starts_with("300")
            threshold = pl.when(is_star_chi).then(0.19).otherwise(0.095)
            
            if 'return_from_pre_close' not in df_minute.columns:
                df_minute = df_minute.with_columns(
                    ((pl.col('last_price') - pl.col('pre_close')) / pl.col('pre_close')).alias('return_from_pre_close')
                )
            
            # 判断当天这只股票实际上是否在某个时刻打板
            df_target = df_minute.group_by("万得代码").agg([
                ((pl.col('return_from_pre_close').max() >= threshold) & (pl.col("touched_limit_up").max() > 0)).alias("hits_limit_up_today")
            ])
            
            df_minute = df_minute.join(df_target, on="万得代码", how="left")
            
            all_intraday_features.append(df_minute)
            
            # --- [V22] 阶段 1优化: 日内极速撮合系统 (Intraday Matcher) ---
            # 严格收窄触发边界：要求极深的流动性真空、更强的买方斜率，且股价至少已启动上涨(>2%)
            df_signals = df_minute.filter(
                (pl.col("bid_ask_slope_diff") > 1.0) & 
                (pl.col("liquidity_void_rate_mean") > 0.85) & 
                (pl.col("return_from_pre_close") > 0.02) & 
                (pl.col("return_from_pre_close") < threshold) # 未封板前
            )
            
            if df_signals.height > 0:
                print(f"    [Trigger] 日内点火器捕捉到 {df_signals.height} 次极速抽单/流动性真空瞬时突变！")
                # 简单统计模拟买点: 假设信号出现时我们在 last_price 买入 (过滤停牌价格)
                df_trades = df_signals.filter(pl.col("last_price") > 0).group_by("万得代码").first()
                if df_trades.height > 0:
                    # 计算到收盘的毛利 (收盘价 / 买入价 - 1)
                    df_close = df_minute.group_by("万得代码").last()
                    df_pnl = df_trades.join(df_close, on="万得代码", suffix="_close")
                    df_pnl = df_pnl.with_columns(
                        ((pl.col("last_price_close") / (pl.col("last_price") + 1e-6)) - 1.0).alias("sim_pnl")
                    )
                    win_trades = df_pnl.filter(pl.col("sim_pnl") > 0).height
                    avg_pnl = df_pnl["sim_pnl"].mean() * 100
                    
                    # 分析是否能封板吃溢价
                    limit_hits = df_pnl.filter(pl.col("touched_limit_up") > 0).height
                    print(f"    [Emulation] 成功以微观信号拦截 {df_pnl.height} 笔, 盈利 {win_trades} 笔 | 日内封板: {limit_hits} 个 | 均空间: {avg_pnl:.2f}%")
                else:
                    print(f"    [Emulation] 信号存在，但未能获取有效价格进行拦截。")

            
        except Exception as e:
            print(f"  -> 处理发生异常: {e}")
            import traceback
            traceback.print_exc()
            
    if all_intraday_features:
        final_intraday_df = pl.concat(all_intraday_features)
        final_intraday_df.write_parquet(INTRADAY_TRAIN_DATA)
        print(f"\n[DONE] MVP 生成完成！日内高频时序截面已保存至: {INTRADAY_TRAIN_DATA}")
        print(f"   共计 {final_intraday_df.height} 条分钟级点火记录。")
    else:
        print("\n未能提取到任何有效的 Intraday 数据！")

if __name__ == "__main__":
    # 为了演示，只测试最后 1 个拥有 Watchlist 的交易日
    test_dates = None
    if os.path.exists(WATCHLIST_PATH):
        dates = pl.read_csv(WATCHLIST_PATH)["date"].unique().to_list()
        dates.sort()
        test_dates = dates[-1:] if len(dates) > 1 else dates
    
    run_intraday_ignition_mvp(test_dates)
