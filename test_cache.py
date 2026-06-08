import polars as pl
import os

OUTPUT_DIR = 'models'
eco_path = 'models/ecosystem_features.parquet'
df_eco = pl.read_parquet(eco_path) if os.path.exists(eco_path) else None

dates = [d.replace('.parquet','') for d in sorted(os.listdir('models/daily_cache')) if d.endswith('.parquet')]

print("Testing Cache Load...")
for date_str in dates[:40]:
    try:
        df_cache = pl.read_parquet(f'models/daily_cache/{date_str}.parquet')
        if df_cache.height > 0:
            toxic_exprs = []
            if "vpin" in df_cache.columns and "intraday_momentum" in df_cache.columns and "toxic_breakout_trap" not in df_cache.columns:
                toxic_exprs.append((pl.col("vpin") / (pl.col("intraday_momentum").abs() + 1e-9)).alias("toxic_breakout_trap"))
            if "limit_up_magnet_effect" in df_cache.columns and "smc_fvg_bearish" in df_cache.columns and "false_magnet_drop" not in df_cache.columns:
                toxic_exprs.append((pl.col("limit_up_magnet_effect") * pl.col("smc_fvg_bearish")).alias("false_magnet_drop"))
            if toxic_exprs:
                df_cache = df_cache.with_columns(toxic_exprs)
            
            # [V21] 合并当日生态学标签
            if df_eco is not None:
                df_eco_day = df_eco.filter(pl.col("date") == int(date_str)).drop("date")
                if df_eco_day.height > 0:
                    df_cache = df_cache.join(df_eco_day, on="万得代码", how="left").fill_null(0.0)
            print(f"{date_str} OK")
    except Exception as e:
        print(f"{date_str} ERR: {e}")
