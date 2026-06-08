import polars as pl
import os
import sys
import time
from engine import compute_and_aggregate_ashare
from tick_factors import compute_tick_factors_for_date
from ml_pipeline import (
    generate_labels_v3, generate_rolling_features, 
    train_ensemble_v3, predict_ensemble_top3, V3_FEATURES
)
from feature_analysis import analyze_feature_importance
from backtest_simulation import simulate_v2_realistic_backtest

SNAPSHOT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "l2_snapshot")
TICK_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "l2_tick")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LGB_PATH = os.path.join(_SCRIPT_DIR, "models", "v3_lgb.txt")


def get_latest_date(base_dir: str) -> str:
    """自动获取最新的可用交易日"""
    dates = sorted(
        [d.replace("date=", "") for d in os.listdir(base_dir) if d.startswith("date=")],
        reverse=True
    )
    return dates[0] if dates else ""


def main(target_date: str = ""):
    """
    A 股 L2 强势打板 AI 系统 V3 入口
    
    Args:
        target_date: 指定日期 (YYYYMMDD)，留空则自动使用最新日期
    """
    if not target_date:
        target_date = get_latest_date(SNAPSHOT_BASE)
        if not target_date:
            print("[ERROR] 未找到任何可用的交易日数据。")
            return
    
    print("=" * 70)
    print(f" A股 L2 强势打板 AI 系统 V3 — 日期: {target_date}")
    print("=" * 70)
    
    data_dir = os.path.join(SNAPSHOT_BASE, f"date={target_date}")
    if not os.path.isdir(data_dir):
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        return
    
    parquet_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
    
    # === 阶段一: Snapshot 因子计算 ===
    start_t = time.time()
    print(f"\n[1] 正在读取 {len(parquet_files)} 个 Snapshot Parquet 文件...")
    df_raw = pl.read_parquet(parquet_files)
    print(f"    原始数据: {df_raw.height} 行")
    
    print(f"\n[2] Snapshot 因子计算 + 60秒聚合...")
    df_agg = compute_and_aggregate_ashare(df_raw, time_step_seconds=60)
    df_agg = generate_rolling_features(df_agg)
    
    # === 阶段二: Tick 因子融合 ===
    print(f"\n[3] 逐笔成交 (Tick) 因子提取与融合...")
    df_tick = compute_tick_factors_for_date(target_date, TICK_BASE)
    if df_tick.height > 0:
        tick_cols = [c for c in df_tick.columns if c not in ("万得代码", "period_start")]
        df_agg = df_agg.join(df_tick, on=["万得代码", "period_start"], how="left")
        df_agg = df_agg.with_columns([pl.col(c).fill_null(0.0) for c in tick_cols])
        print(f"    Tick 因子融合完成! 新增 {len(tick_cols)} 列")
    
    print(f"\n    V3 聚合宽表: {df_agg.height} 行 x {len(df_agg.columns)} 列")
    print(f"    阶段一+二 总耗时: {time.time() - start_t:.1f}s")
    
    # === 阶段三: 截面截取 + 标签 ===
    print("\n============== [阶段三: AI 三模型集成训练] ==============")
    df_last = (
        df_agg.sort("period_start")
        .group_by("万得代码").last()
    )
    
    df_with_labels = generate_labels_v3(df_last, tp_target=0.10, max_dd=-0.03)
    
    # === 阶段四: 三模型训练 ===
    lgb_m, xgb_m, cb_m = train_ensemble_v3(df_with_labels, features=V3_FEATURES)
    
    # === 阶段五: 特征重要性分析 ===
    print("\n============== [阶段四: 特征重要性分析] ==============")
    kept_features = analyze_feature_importance(LGB_PATH, V3_FEATURES)
    
    # === 阶段六: 集成投票推选 Top 3 ===
    df_top3 = predict_ensemble_top3(df_last, features=V3_FEATURES)
    
    print("\n============== 【V3 每日策略输出: 三模型投票 TOP 3】 ==============")
    print(df_top3)
    
    # === 阶段七: 模拟回测 ===
    simulate_v2_realistic_backtest(df_top3, df_with_labels)

if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    main(target_date=date_arg)
