import polars as pl
import os
import glob
from datetime import datetime

print("Loading strategy... ")
df_strategy = pl.read_csv('models/wf_equity_curve.csv')
df_strategy = df_strategy.with_columns(
    pl.col('date').cast(pl.Utf8).str.slice(0, 6).alias('month')
)

# 计算每个月末的累积净值 / 上月最后一天净值 - 1 = 本月真实盈亏
# 由于 dataframe 可能并不是完全从月初第一天开始
def calc_strat_month_ret(df: pl.DataFrame):
    months = df.select("month").unique().sort("month").to_series().to_list()
    last_cap = 1000000.0  # 假设初始资金 100万元
    res = []
    for m in months:
        df_m = df.filter(pl.col("month") == m).sort("date")
        end_cap = df_m.select("capital").tail(1).item()
        res.append({"month": m, "strategy_ret": (end_cap / last_cap) - 1.0})
        last_cap = end_cap
    return pl.DataFrame(res)

strategy_monthly = calc_strat_month_ret(df_strategy)

# 2. 计算基准 (等权每日计算)
print("Loading market data...")
cache_files = glob.glob('models/daily_cache/*.parquet')
market_daily = []

for file in sorted(cache_files):
    date_str = os.path.basename(file).split('_')[0]
    if not date_str.isdigit() or len(date_str) != 8:
        continue
    # 只取我们需要的时间区间
    if date_str < df_strategy.select("date").cast(pl.Utf8).min() or date_str > df_strategy.select("date").cast(pl.Utf8).max():
        continue
        
    try:
        df_cache = pl.read_parquet(file, columns=['万得代码', 'pre_close', 'last_price'])
        # 排除停牌和一字跌停、涨停的极端无效价格
        df_ret = df_cache.filter(pl.col('pre_close') > 0).with_columns([
            ((pl.col('last_price') - pl.col('pre_close')) / pl.col('pre_close')).alias('ret')
        ])
        if df_ret.height > 0:
            avg_ret = df_ret.select(pl.col('ret').mean()).item()
            market_daily.append({'date': date_str, 'month': date_str[:6], 'market_daily_ret': avg_ret})
    except Exception as e:
        continue

if market_daily:
    df_market = pl.DataFrame(market_daily)
    # 按月聚合并计算复利
    market_monthly = (
        df_market.with_columns(
            (pl.col('market_daily_ret') + 1.0).alias('gross_ret')
        )
        .group_by('month').agg([
            (pl.col('gross_ret').product() - 1.0).alias('market_ret')
        ])
    )
else:
    market_monthly = None

if market_monthly is not None:
    df_merged = strategy_monthly.join(market_monthly, on='month', how='inner').sort('month')
    df_merged = df_merged.with_columns([
        (pl.col('strategy_ret') - pl.col('market_ret')).alias('alpha')
    ])
    
    print("\n| 月份 | 策略收益 | 大盘(等权) | 超额收益 (Alpha) |")
    print("|------|----------|------------|------------------|")
    for row in df_merged.iter_rows(named=True):
        m = row['month']
        s = row['strategy_ret'] * 100
        mk = row['market_ret'] * 100
        a = row['alpha'] * 100
        print(f"| {m[:4]}-{m[4:]} | {s:6.2f}% | {mk:7.2f}% | {a:9.2f}% |")

    strat_cum = (1 + df_merged.select(pl.col('strategy_ret'))).product().item() - 1
    mkt_cum = (1 + df_merged.select(pl.col('market_ret'))).product().item() - 1
    
    print("\n**总结 (Over Matched Months)**")
    print(f"- **策略累计收益**: {strat_cum*100:.2f}%")
    print(f"- **大盘累计收益**: {mkt_cum*100:.2f}%")
    print(f"- **累计超额 Alpha**: {(strat_cum - mkt_cum)*100:.2f}%")
else:
    print("Cannot compute market benchmark.")
