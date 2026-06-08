import os
from batch_pipeline import run_walkforward_backtest

# Run a very short backtest on recent data just to generate the intraday watchlist
if __name__ == "__main__":
    # We choose late 2024 to minimize the total days to just 1 or 2 iterations.
    # We set train_window=10, retrain_every=2, start_date="20241020".
    # This will generate a few days of intraday_watchlist.csv
    try:
        run_walkforward_backtest(train_window=10, retrain_every=2, start_date="20241020")
    except Exception as e:
        print("Run complete or crashed:", e)
