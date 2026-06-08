import polars as pl
from batch_pipeline import daily_cache_to_features
from ml_pipeline import predict_ensemble_top3
import json

file_path = 'models/daily_cache/20240626.parquet'
df_cache = pl.read_parquet(file_path)

df_ret = df_cache.filter(pl.col('pre_close') > 0).with_columns([
    ((pl.col('last_price') - pl.col('pre_close')) / pl.col('pre_close')).alias('ret')
])
print(f'Median Market Return on 20240626: {df_ret.select(pl.col("ret").median()).item()*100:.2f}%')

df_test = daily_cache_to_features([file_path], 'models')
df_test_filtered = df_test.filter(~((pl.col('last_price') == 0.0) | pl.col('万得代码').str.contains('ST')))

with open('models/all_features.json', 'r') as f:
    all_features = json.load(f)

# The ML models state for 20240626 should ideally be the ones right before that day,
# but the current trained models in `models/` folder are at whatever the last step of the backtest is (2025-03).
# Still, they can give us a general sense of which stocks were rated highly on 20240626.
df_top3 = predict_ensemble_top3(df_test_filtered, all_features)

print('--- Top 3 Stocks Evaluated on 20240626 ---')
print(df_top3.select(['万得代码', 'ensemble_avg', 'big_net_inflow', 'vpin']))
