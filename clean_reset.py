"""
全新启动回测前的清理脚本：删除旧断点、日志和模型状态文件
"""
import os
import glob

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

FILES_TO_REMOVE = [
    # WF 断点续传日志
    os.path.join(MODELS_DIR, "wf_pnl.log"),
    os.path.join(MODELS_DIR, "wf_pnl_top3.log"),
    # 净值曲线
    os.path.join(MODELS_DIR, "wf_equity_curve.csv"),
    os.path.join(MODELS_DIR, "wf_equity_curve_top3.csv"),
    # 崩溃日志
    os.path.join(MODELS_DIR, "wf_crash_error.log"),
    os.path.join(MODELS_DIR, "oos_crash_error.log"),
    # 因子生命周期
    os.path.join(MODELS_DIR, "importance_history.csv"),
    os.path.join(MODELS_DIR, "dead_factors.json"),
    os.path.join(MODELS_DIR, "discovered_factors.json"),
    os.path.join(MODELS_DIR, "active_features.json"),
    os.path.join(MODELS_DIR, "feature_importance.json"),
    # 项目根目录杂项
    os.path.join(PROJECT_DIR, "run_wf.log"),
    os.path.join(PROJECT_DIR, "temp_error.log"),
]

removed = 0
for f in FILES_TO_REMOVE:
    if os.path.exists(f):
        os.remove(f)
        print(f"  [DEL] {os.path.basename(f)}")
        removed += 1

print(f"\n清理完成: 删除了 {removed} 个旧文件，断点已重置。")
print("现在可以运行: python start_wf.py")
