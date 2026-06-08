import polars as pl
import json

file_path = 'models/daily_cache/20240903.parquet'
df = pl.read_parquet(file_path, columns=['万得代码', 'pre_close', 'last_price', 'big_net_inflow', 'vpin'])
df_ret = df.filter(pl.col('pre_close') > 0).with_columns([
    ((pl.col('last_price') - pl.col('pre_close')) / pl.col('pre_close')).alias('ret')
])
df_trap = df_ret.filter(pl.col('ret') > 0.08).filter(pl.col('big_net_inflow') < 0).sort('ret', descending=True).head(3)
print('--- 20240903 Traps ---')
for row in df_trap.iter_rows(named=True):
    print("{} | ret: {:.2f}% | Inflow: {:.2f} | VPIN: {:.2f}".format(row['万得代码'], row['ret']*100, row['big_net_inflow'], row['vpin']))

print('--- 20240626 Event ---')
df2 = pl.read_parquet('models/daily_cache/20240626.parquet', columns=['万得代码', 'pre_close', 'last_price', 'big_net_inflow', 'vpin'])
df_ret2 = df2.filter(pl.col('pre_close') > 0).with_columns([
    ((pl.col('last_price') - pl.col('pre_close')) / pl.col('pre_close')).alias('ret')
])
df_top2 = df_ret2.sort('ret', descending=True).head(5)
for row in df_top2.iter_rows(named=True):
    print("{} | ret: {:.2f}% | Inflow: {:.2f} | VPIN: {:.2f}".format(row['万得代码'], row['ret']*100, row['big_net_inflow'], row['vpin']))
