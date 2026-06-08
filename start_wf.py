import sys
import os

# 将当前目录添加到 PYTHONPATH，以便正确导入 batch_pipeline 等模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from batch_pipeline import run_walkforward_backtest

if __name__ == "__main__":
    print("Starting Walk-Forward Backtest with resume logic...")
    # 可以传递参数，这里默认使用默认参数，它会自动寻找 models/wf_pnl.log 实现断点续传
    run_walkforward_backtest()
