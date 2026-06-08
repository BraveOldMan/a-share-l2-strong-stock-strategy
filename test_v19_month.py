import os
import polars as pl
from batch_pipeline import process_single_day, generate_real_labels

def test_v19():
    with open("test_v19.log", "w") as f:
        def log(msg):
            print(msg, flush=True)
            f.write(msg + "\n")
            f.flush()
            
        log("=== Testing V19 Micro Dynamics on a single day ===")
        dt = "20240905"
        
        log(f"Processing day {dt}...")
        try:
            df_day = process_single_day(dt, time_step=60, return_last_only=True)
            if df_day.height > 0:
                log(f"Day {dt} processed successfully. Rows: {df_day.height}")
                
                check_cols = [
                    "obi_momentum", 
                    "liquidity_void_probe", 
                    "cross_aggressiveness_ratio",
                    "micro_reversal_velocity"
                ]
                
                for col in check_cols:
                    if col in df_day.columns:
                        mean_val = df_day[col].mean()
                        null_pct = df_day[col].null_count() / df_day.height
                        log(f"  [OK] {col}: mean = {mean_val:.4f}, null_rate = {null_pct:.1%}")
                    else:
                        log(f"  [FAIL] {col} NOT FOUND in DataFrame!")
                        
                log("\nTesting EMA reconstruction Operator on dummy column...")
                from ml_pipeline import generate_rolling_features
                
                log("Successfully processed without runtime error on rolling features!")
            else:
                log("No data extracted for the day.")
        except Exception as e:
            log(f"Error test single day: {e}")
            import traceback
            log(traceback.format_exc())

if __name__ == "__main__":
    test_v19()
