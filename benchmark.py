import polars as pl
import os

OUTPUT_DIR = "models"
start_date = "20240605"
end_date = "20240703"

path_start = os.path.join(OUTPUT_DIR, "daily_cache", f"{start_date}.parquet")
path_end = os.path.join(OUTPUT_DIR, "daily_cache", f"{end_date}.parquet")

if not os.path.exists(path_start) or not os.path.exists(path_end):
    print("Cache files not found")
else:
    df_start = pl.read_parquet(path_start).sort("period_start").group_by("万得代码").last().select(["万得代码", pl.col("last_price").alias("price_start")])
    df_end = pl.read_parquet(path_end).sort("period_start").group_by("万得代码").last().select(["万得代码", pl.col("last_price").alias("price_end")])
    
    df = df_start.join(df_end, on="万得代码", how="inner")
    df = df.filter((pl.col("price_start") > 0) & (pl.col("price_end") > 0))
    df = df.with_columns(((pl.col("price_end") / pl.col("price_start")) - 1).alias("return"))
    
    print(f"Benchmark Return ({start_date} to {end_date}):")
    print(f"Universe Size: {df.height}")
    print(f"Equal-Weight Mean Return: {df.select(pl.col('return').mean()).item() * 100:.2f}%")
    print(f"Median Return: {df.select(pl.col('return').median()).item() * 100:.2f}%")
    
    # Let's also count how many dropped more than 10%, 20%
    drop_10 = df.filter(pl.col('return') <= -0.10).height
    drop_20 = df.filter(pl.col('return') <= -0.20).height
    print(f"Stocks dropping > 10%: {drop_10} ({(drop_10/df.height)*100:.1f}%)")
    print(f"Stocks dropping > 20%: {drop_20} ({(drop_20/df.height)*100:.1f}%)")
