import os
import sys
import multiprocessing

# Change to the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 延迟导入：必须在 __main__ 保护块内，否则 Windows spawn 子进程会递归执行
    import batch_pipeline
    batch_pipeline.run_walkforward_backtest(100, 20)
