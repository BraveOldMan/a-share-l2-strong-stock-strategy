import polars as pl
import os

OUTPUT_DIR = "models"

def test_dist():
    dates = ["20240702", "20240703", "20240704"]
    dfs = []
    for d in dates:
        path = os.path.join(OUTPUT_DIR, "daily_cache", f"{d}.parquet")
        if os.path.exists(path):
            dfs.append(pl.read_parquet(path))
            
    df_input = pl.concat(dfs, how="diagonal_relaxed")
    
    df_daily = df_input.sort("period_start").group_by(["万得代码", "trade_date"]).last()
    
    if "day_high" in df_input.columns:
        df_hilo = df_daily.select(["万得代码", "trade_date", "day_open", "day_high", "day_low", "day_close"])
    else:
        df_hilo = df_daily.with_columns(pl.col("last_price").alias("day_high"))  # simplify
        
    dates_list = sorted(df_daily.select("trade_date").unique().to_series().to_list())
    for i, dt in enumerate(dates_list):
        df_today = df_daily.filter(pl.col("trade_date") == dt)
        if i + 2 < len(dates_list):
            next_dt = dates_list[i + 1]
            next_next_dt = dates_list[i + 2]
            
            df_next_hilo = df_hilo.filter(pl.col("trade_date") == next_dt).select([
                pl.col("万得代码"),
                pl.col("day_open").alias("next_day_open"),
                pl.col("day_low").alias("next_day_low"),
                pl.col("day_close").alias("next_day_close"),
            ])
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
                  .otherwise(0.0).alias("future_max_return")
            )
            print(f"\n--- {dt} Labeled Data ---")
            print(df_labeled.select(["万得代码", "trade_date", "next_day_open", "next_next_day_high", "future_max_return"]).head(10))
            
if __name__ == "__main__":
    test_dist()
