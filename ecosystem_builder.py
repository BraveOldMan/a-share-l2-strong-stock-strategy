import os
import glob
import polars as pl
import pandas as pd
from tqdm import tqdm

DAILY_CACHE_DIR = r"d:\量化策略\Factor Generation\A股L2强势股策略\models\daily_cache"
OUTPUT_PARQUET = r"d:\量化策略\Factor Generation\A股L2强势股策略\models\ecosystem_features.parquet"

def build_ecosystem():
    files = glob.glob(os.path.join(DAILY_CACHE_DIR, "*.parquet"))
    # Sort files chronologically based on filename (YYYYMMDD.parquet)
    files.sort(key=lambda x: os.path.basename(x))
    
    if not files:
        print("No daily cache parquet files found!")
        return
        
    print(f"Found {len(files)} daily cache files. Building ecosystem...")
    
    # State tracking: {symbol: consecutive_limit_ups}
    consecutive_tracker = {}
    
    results = []
    
    for f in tqdm(files):
        date_str = os.path.basename(f).replace(".parquet", "")
        
        try:
            df = pl.read_parquet(f)
            
            # Identify 20% limit boards (300* or 688*) vs 10% limit boards
            is_20_pct = pl.col('万得代码').str.starts_with("300") | pl.col('万得代码').str.starts_with("688")
            
            # Simple heuristic for limit up detection based on daily cache returns
            threshold = pl.when(is_20_pct).then(0.19).otherwise(0.095)
            
            # Compute limit up and broken board
            if 'return_from_pre_close' not in df.columns:
                df = df.with_columns(((pl.col('last_price') - pl.col('pre_close')) / pl.col('pre_close')).alias('return_from_pre_close'))
            
            df = df.with_columns([
                ((pl.col('return_from_pre_close') >= threshold) & (pl.col('touched_limit_up') > 0)).alias("is_limit_up_close"),
                ((pl.col('touched_limit_up') > 0) & (pl.col('return_from_pre_close') < threshold)).alias("is_broken_board")
            ])
            
            # Extract lists of symbols
            symbols = df['万得代码'].to_list()
            is_lu_list = df['is_limit_up_close'].to_list()
            is_broken_list = df['is_broken_board'].to_list()
            
            day_records = []
            
            # Global limit up stats for today
            today_lu_count = sum(is_lu_list)
            yesterday_1_board_symbols = [s for s, count in consecutive_tracker.items() if count == 1]
            yesterday_2plus_board_symbols = [s for s, count in consecutive_tracker.items() if count >= 2]
            
            promotion_1_to_2 = 0
            promotion_2_plus = 0
            
            # Update state machine and build records
            for sym, is_lu, is_broken in zip(symbols, is_lu_list, is_broken_list):
                if is_lu:
                    consecutive_tracker[sym] = consecutive_tracker.get(sym, 0) + 1
                else:
                    consecutive_tracker[sym] = 0
                    
                consecutive_count = consecutive_tracker[sym]
                
                if is_lu:
                    if sym in yesterday_1_board_symbols:
                        promotion_1_to_2 += 1
                    if sym in yesterday_2plus_board_symbols:
                        promotion_2_plus += 1
                
                day_records.append({
                    "date": int(date_str),
                    "万得代码": sym,
                    "consecutive_limit_ups": consecutive_count,
                    "is_broken_board": 1 if is_broken else 0,
                })
                
            # Global features representing market sentiment
            market_promotion_rate_1_to_2 = promotion_1_to_2 / max(1, len(yesterday_1_board_symbols))
            market_highest_space_board = max(consecutive_tracker.values()) if consecutive_tracker else 0
            
            for rec in day_records:
                rec["market_limit_up_count_eco"] = today_lu_count
                rec["market_promotion_rate_1_to_2"] = market_promotion_rate_1_to_2
                rec["market_highest_space_board"] = market_highest_space_board
                
            results.extend(day_records)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    # Combine and save
    print("Combining results into parquet...")
    df_results = pl.DataFrame(results)
    
    # We need to shift the ecosystem features by 1 day because they are "today's" stats.
    # The machine learning model standing on Day T uses Day T's features to predict T+1.
    # Therefore, Day T's ecosystem is perfectly aligned. No need to shift manually here if we strict merge on date.
    
    df_results.write_parquet(OUTPUT_PARQUET)
    print(f"Ecosystem features successfully written to: {OUTPUT_PARQUET}")

if __name__ == "__main__":
    build_ecosystem()
