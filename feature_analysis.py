import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LGB_PATH = os.path.join(MODEL_DIR, "v3_lgb.txt")
XGB_PATH = os.path.join(MODEL_DIR, "v3_xgb.ubj")
CB_PATH = os.path.join(MODEL_DIR, "v3_catboost")


def _get_lgb_importance(features: list) -> np.ndarray:
    """获取 LightGBM 特征重要性"""
    if not os.path.exists(LGB_PATH):
        return np.zeros(len(features))
    gbm = lgb.Booster(model_file=LGB_PATH)
    return gbm.feature_importance(importance_type='gain').astype(np.float64)


def _get_xgb_importance(features: list) -> np.ndarray:
    """获取 XGBoost 特征重要性"""
    if not os.path.exists(XGB_PATH):
        return np.zeros(len(features))
    bst = xgb.Booster()
    bst.load_model(XGB_PATH)
    score_map = bst.get_score(importance_type='gain')
    # XGBoost 用 f0, f1... 命名，需要映射回原始特征名
    imp = np.zeros(len(features))
    for i, f in enumerate(features):
        imp[i] = score_map.get(f, score_map.get(f"f{i}", 0.0))
    return imp


def _get_cb_importance(features: list) -> np.ndarray:
    """获取 CatBoost 特征重要性"""
    if not os.path.exists(CB_PATH):
        return np.zeros(len(features))
    model = cb.CatBoostClassifier()
    model.load_model(CB_PATH)
    return model.get_feature_importance().astype(np.float64)


def analyze_feature_importance(model_path: str, features: list) -> list:
    """
    三模型共识特征重要性分析：
    对 LightGBM、XGBoost、CatBoost 的 Gain 重要性进行归一化加权平均，
    输出三模型共识排名并自动筛选重要特征。
    """
    print("\n============== [V3 三模型共识特征重要性排名] ==============")
    
    imp_lgb = _get_lgb_importance(features)
    imp_xgb = _get_xgb_importance(features)
    imp_cb = _get_cb_importance(features)
    
    # 归一化 (各模型内部 Min-Max)
    def _normalize(arr: np.ndarray) -> np.ndarray:
        total = arr.sum()
        return arr / total if total > 0 else arr
    
    norm_lgb = _normalize(imp_lgb)
    norm_xgb = _normalize(imp_xgb)
    norm_cb = _normalize(imp_cb)
    
    # 等权平均 (三模型共识)
    consensus = (norm_lgb + norm_xgb + norm_cb) / 3.0
    
    # 排名
    feat_imp = list(zip(features, consensus, norm_lgb, norm_xgb, norm_cb))
    feat_imp.sort(key=lambda x: x[1], reverse=True)
    
    total_consensus = consensus.sum() + 1e-9
    kept_features = []
    
    print(f"{'Rank':>4s}  {'Tag':>6s}  {'Feature':30s} | {'Consensus':>10s}  {'LGB':>6s}  {'XGB':>6s}  {'CB':>6s}")
    print("-" * 85)
    
    for rank, (fname, cons, lg, xg, cb_v) in enumerate(feat_imp, 1):
        pct = cons / total_consensus * 100
        tag = "[KEEP]" if pct >= 1.0 else "[DROP]"
        print(f"  #{rank:2d}  {tag}  {fname:30s} | {pct:9.1f}%  {lg:5.1%}  {xg:5.1%}  {cb_v:5.1%}")
        if pct >= 1.0:
            kept_features.append(fname)
    
    print(f"\n[Feature] 原始特征数: {len(features)} -> 筛选后: {len(kept_features)}")
    return kept_features
