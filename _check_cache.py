import os

dates = sorted([d.replace('date=','') for d in os.listdir('E:/AGUDATA/l2_snapshot') if d.startswith('date=')])
print(f"Total snapshot days: {len(dates)}")
print(f"First: {dates[0]}")
print(f"Last: {dates[-1]}")

cached = sorted([f.replace('.parquet','') for f in os.listdir('models/daily_cache') if f.endswith('.parquet')])
missing = [d for d in dates if d not in cached]

print(f"\nCached: {len(cached)} days")
print(f"Missing from cache: {len(missing)} days")
if missing:
    print(f"Missing range: {missing[0]} ~ {missing[-1]}")
    # Show first 10 missing
    print(f"First 10 missing: {missing[:10]}")
