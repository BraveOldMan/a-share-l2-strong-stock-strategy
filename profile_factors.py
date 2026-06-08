import polars as pl
import time
import numpy as np
from engine import (
    format_datetime, assign_session_id
)
from custom_factors import (
    factor_ofi_multi_level_ashare, factor_volatility_ashare,
    factor_simple_obi_ashare, factor_weighted_price_diff_ashare,
    factor_depth_variance, factor_book_slope
)
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def profile_compute(df_combined):
    df_combined = df_combined.head(10000) # Ensure it doesn't take too long if it's crazy
    t0 = time.time()
    
    t_start = time.time()
    df_combined = format_datetime(df_combined)
    logging.info(f"format_datetime: {time.time() - t_start:.4f}s")
    
    t_start = time.time()
    price_cols = [f'申买价{i}' for i in range(1, 11)] + [f'申卖价{i}' for i in range(1, 11)] + ['pre_close', 'last_price']
    df_combined = df_combined.with_columns(
        [(pl.col(c) / 10000.0).cast(pl.Float32) for c in price_cols if c in df_combined.columns]
    )
    logging.info(f"div 10000: {time.time() - t_start:.4f}s")
    
    t_start = time.time()
    df_combined = assign_session_id(df_combined)
    df_combined = df_combined.sort(["datetime", "万得代码"])
    logging.info(f"assign_session_id & sort: {time.time() - t_start:.4f}s")
    
    t_start = time.time()
    bp_cols = [f'申买价{i}' for i in range(1, 11)]
    ba_cols = [f'申买量{i}' for i in range(1, 11)]
    ap_cols = [f'申卖价{i}' for i in range(1, 11)]
    aa_cols = [f'申卖量{i}' for i in range(1, 11)]
    mat_bp = df_combined.select(bp_cols).to_numpy() if all(c in df_combined.columns for c in bp_cols) else np.zeros((df_combined.height, 10))
    mat_ba = df_combined.select(ba_cols).to_numpy() if all(c in df_combined.columns for c in ba_cols) else np.zeros((df_combined.height, 10))
    mat_ap = df_combined.select(ap_cols).to_numpy() if all(c in df_combined.columns for c in ap_cols) else np.zeros((df_combined.height, 10))
    mat_aa = df_combined.select(aa_cols).to_numpy() if all(c in df_combined.columns for c in aa_cols) else np.zeros((df_combined.height, 10))
    logging.info(f"to_numpy: {time.time() - t_start:.4f}s")

    mid_prices = (mat_bp[:, 0] + mat_ap[:, 0]) / 2.0
    ask1_zeros = (mat_ap[:, 0] == 0.0)
    mid_prices[ask1_zeros] = mat_bp[ask1_zeros, 0] 
    
    t_start = time.time()
    ofi_multi = factor_ofi_multi_level_ashare(mat_bp, mat_ba, mat_ap, mat_aa, levels=10)
    logging.info(f"Numba ofi_multi: {time.time() - t_start:.4f}s")

    t_start = time.time()
    volatility = factor_volatility_ashare(mid_prices, window=20)
    logging.info(f"Numba volatility: {time.time() - t_start:.4f}s")
    
    t_start = time.time()
    price_diff = factor_weighted_price_diff_ashare(mat_bp, mat_ba, mat_ap, mat_aa, levels=5)
    logging.info(f"Numba price_diff: {time.time() - t_start:.4f}s")
    
    t_start = time.time()
    depth_var = factor_depth_variance(mat_ba, mat_aa, levels=10)
    logging.info(f"Numba depth_var: {time.time() - t_start:.4f}s")
    
    t_start = time.time()
    book_slope = factor_book_slope(mat_ba, mat_aa, levels=10)
    logging.info(f"Numba book_slope: {time.time() - t_start:.4f}s")

    t_start = time.time()
    valid_ba = [c for c in ba_cols if c in df_combined.columns] or [pl.lit(0.0)]
    valid_aa = [c for c in aa_cols if c in df_combined.columns] or [pl.lit(0.0)]
    df_combined = df_combined.with_columns(
        pl.sum_horizontal(valid_ba).alias("bid_vols_10"),
        pl.sum_horizontal(valid_aa).alias("ask_vols_10")
    )
    df_combined = df_combined.with_columns(
        ((pl.col("bid_vols_10") - pl.col("ask_vols_10")) / (pl.col("bid_vols_10") + pl.col("ask_vols_10") + 1e-9)).alias("oir_10")
    )
    logging.info(f"Polars Pre-calc (bid/ask vols): {time.time() - t_start:.4f}s")

    t_start = time.time()
    depth_impact = np.zeros(len(mat_ap), dtype=np.float32)
    for row_i in range(len(mat_ap)):
        cum_vol = 0.0
        cum_amt = 0.0
        for lvl in range(10):
            p, v = mat_ap[row_i, lvl], mat_aa[row_i, lvl]
            if p <= 0 or v <= 0:
                continue
            cum_vol += v
            cum_amt += p * v
        if cum_vol > 0 and mat_ap[row_i, 0] > 0:
            depth_impact[row_i] = (cum_amt / cum_vol - mat_ap[row_i, 0]) / (mat_ap[row_i, 0] + 1e-9)
    logging.info(f"Python Depth Impact for loop: {time.time() - t_start:.4f}s")

    t_start = time.time()
    cols_to_select = ["万得代码", "datetime", "session_id"]
    for c, alias in [("申卖价1", "ap1"), ("申买价1", "bp1"), ("申买量1", "bv1"), ("申卖量1", "av1")]:
       if c in df_combined.columns:
           cols_to_select.append(pl.col(c).alias(alias))
       else:
           cols_to_select.append(pl.lit(0.0).alias(alias))
    cols_to_select.extend(["bid_vols_10", "ask_vols_10", "oir_10"])
    for c in ["pre_close", "last_price", "当日累计成交量", "品种总数"]:
       if c in df_combined.columns:
           cols_to_select.append(pl.col(c))
       else:
           cols_to_select.append(pl.lit(0.0).alias(c))

    df_features = df_combined.select(cols_to_select).with_columns([
        pl.Series("factor_ofi_10", ofi_multi),
        pl.Series("factor_volatility", volatility),
        pl.Series("factor_price_diff", price_diff),
        pl.Series("factor_depth_var", depth_var),
        pl.Series("factor_book_slope", book_slope),
        (pl.col("ap1") == 0.0).cast(pl.Int32).alias("is_limit_up"),
        (pl.col("bv1") * pl.col("bp1")).alias("limit_up_order_amt"),
        ((pl.col("ap1") - pl.col("bp1")) / (pl.col("bp1") + 1e-9)).alias("spread_ratio"),
        ((pl.col("bv1") - pl.col("av1")) / (pl.col("bv1") + pl.col("av1") + 1e-9)).alias("bid_ask_imbalance"),
        ((((pl.col("bp1") * pl.col("av1") + pl.col("ap1") * pl.col("bv1")) / (pl.col("bv1") + pl.col("av1") + 1e-9)) / (pl.col("last_price") + 1e-9)) - 1.0).alias("wmp_divergence"),
    ])
    df_features = df_features.with_columns(pl.Series("depth_weighted_impact", depth_impact))
    logging.info(f"Polars Select + With Columns + V9 factors: {time.time() - t_start:.4f}s")
    
    t_start = time.time()
    time_step_seconds = 60
    df_agg = (
        df_features
        .with_columns(pl.col("datetime").dt.truncate(f"{time_step_seconds}s").alias("period_start"))
        .group_by(["万得代码", "period_start"])
        .agg([
            pl.col("bp1").last().alias("close_bp1"),
            pl.col("factor_ofi_10").sum().alias("ofi_sum"),
            pl.col("factor_volatility").mean().alias("volatility_mean"),
            pl.col("factor_price_diff").mean().alias("price_diff_mean"),
        ])
    )
    logging.info(f"Polars Groupby Aggregation (mock small): {time.time() - t_start:.4f}s")
    logging.info(f"TOTAL TIME: {time.time() - t0:.4f}s")

if __name__ == "__main__":
    df = pl.read_parquet('E:/AGUDATA/l2_snapshot/date=20250610/part_0000.parquet')
    profile_compute(df)
