import polars as pl
from ml_pipeline import predict_ensemble_top3
from batch_pipeline import daily_cache_to_features
import json

file_path = 'models/daily_cache/20240903.parquet'
df = pl.read_parquet(file_path)

df_ret = df.filter(pl.col('pre_close') > 0).with_columns([
    ((pl.col('last_price') - pl.col('pre_close')) / pl.col('pre_close')).alias('ret')
])

print('=== 20240903 异常涨停板 (大单净流出 / VPIN 过高) ===')
df_limit = df_ret.filter(pl.col('ret') > 0.09)

# Traps
df_trap = df_limit.filter((pl.col('big_net_inflow') <= 0) | (pl.col('vpin') > 0.8)).sort('ret', descending=True).head(5)
for row in df_trap.iter_rows(named=True):
    print(f"{row['万得代码']} | 涨幅: {row['ret']*100:.2f}% | 大单净流: {row['big_net_inflow']:.2f} | VPIN: {row['vpin']:.2f}")

# Re-run prediction
df_test = daily_cache_to_features([file_path], 'models')
df_test_filtered = df_test.filter(~((pl.col('last_price') == 0.0) | pl.col('万得代码').str.contains('ST')))

with open('models/all_features.json', 'r') as f:
    all_features = json.load(f)

df_top3 = predict_ensemble_top3(df_test_filtered, all_features)
print('\n=== 原本由模型选出的 Top3 高分标的 ===')
if df_top3.height > 0:
    for row in df_top3.iter_rows(named=True):
        print(f"{row['万得代码']} | 绝对得分: {row['ensemble_avg']:.4f} | 净流出(主卖拦截): {row['big_net_inflow']:.4f} | VPIN: {row['vpin']:.4f}")
else:
    print("模型没有选出超过 0.05 得分的票。")
