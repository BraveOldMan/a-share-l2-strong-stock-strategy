import polars as pl
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import os
import tempfile
import shutil

# 恢复到当前项目 D 盘的 models 文件夹
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
LGB_PATH = os.path.join(MODEL_DIR, "v3_lgb.txt")
XGB_PATH = os.path.join(MODEL_DIR, "v3_xgb.ubj")
CB_PATH = os.path.join(MODEL_DIR, "v3_catboost")
TRANSFORMER_PATH = os.path.join(MODEL_DIR, "v4_transformer.pth")
REGIME_PATH = os.path.join(MODEL_DIR, "v5_regime.joblib")

import joblib
from market_regime_gmm import MarketRegimeDetector

try:
    from dl_pipeline import train_transformer, extract_transformer_embeddings, TemporalFeatureExtractor
    import torch
    HAS_DL = True
except ImportError:
    HAS_DL = False

def _safe_load_model(model_lib, load_func, real_path, **kwargs):
    """安全加载模型，避开 C++ 库对中文路径的支持缺陷"""
    if not os.path.exists(real_path):
        return None
    fd, temp_path = tempfile.mkstemp(suffix=".model")
    os.close(fd)
    try:
        shutil.copy2(real_path, temp_path)
        return load_func(temp_path, **kwargs)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def _safe_save_model(save_func, real_path, **kwargs):
    """安全保存模型，避开 C++ 库对中文路径的支持缺陷"""
    fd, temp_path = tempfile.mkstemp(suffix=".model")
    os.close(fd)
    try:
        save_func(temp_path, **kwargs)
        shutil.move(temp_path, real_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

# ===================== V3 特征集 (含 Tick + Session) =====================
V3_FEATURES = [
    # Snapshot 核心
    "pre_close", "ofi_sum", "depth_var_last",
    "volatility_mean", "max_limit_up_order_amt", "close_bp1", "close_ap1",
    # Rolling 时序 (Top Performers)
    "ofi_sum_ma10", "vol_max_10",
    # Tick 逐笔成交因子
    "tick_freq_per_sec",
    # 执行级微观摩擦与偏移因子
    "spread_ratio_mean", "bid_ask_imbalance_mean", "vwap_divergence",
    # 进阶微观结构因子
    "wmp_divergence_mean", "oir_10_mean",
    # 顶流订单流因子 (Order Flow Library)
    # 极限结构因子
    "book_slope_mean", "depth_concentration_mean", "vol_acceleration", "liquidity_void_rate_mean",
    # [V6] Crypto 微观因子
    "kaufman_er", "cvd_divergence",
    # [V7] SMC 聪明钱形态学
    "smc_fvg_bullish", "smc_fvg_bearish",
    # [V7] Cross-Domain 跨域交叉因子
    "cross_trade_order_divergence", "cross_book_depletion_rate",
    # [V9] 涨停特色微观博弈因子 (Limit-up Microstructure)
    # [V10] 前沿微观高频高阶动量与挂撤单
    "vol_volatility_efficiency",
    # [V10.2] Snapshot 侧深度订单流因子
    "depth_weighted_impact_mean",
    # [V10.2] Tick 侧高阶订单流因子
    "trade_size_hhi",
    # [V10.2] Rolling 层订单流加速度与惯性
    "ofi_acceleration", "intraday_momentum",
    # [V10.2] SMC 聪明钱高阶因子
    "smc_premium_discount", "smc_bos", "smc_wyckoff_accum",
    # [V10.2] 冰山/隐藏/拆单检测因子
    "stealth_accumulation", "phantom_liquidity",
    "hidden_fill_ratio", "depth_asym_persistence",
    "consecutive_same_size_ratio", "trade_size_clustering",
    # [V10.2] A股打板巅峰专项因子 (VP & 盘口形状 & 涨停自激)
    "bid_ask_slope_diff",
    "near_limit_up_accumulation",
    "poc_distance", "value_area_breakout", "vp_skewness",
    # [V11] 防守截面因子与尾部风险
    "relative_volume_rank_10m", "intraday_strength_zscore",
    # [V19] 深度盘口动力学组与微观击穿毒性组
    "obi_momentum", "liquidity_void_probe", "cross_aggressiveness_ratio"
]

def generate_labels_v3(df: pl.DataFrame, tp_target=0.15, default_max_dd=-0.05) -> pl.DataFrame:
    """
    V18 排序学习 (Learning-to-Rank) 截面动态标签
    计算逻辑：
    - 直接将 `future_max_return` (次日相对建仓基准的最大涨幅) 取 rank。
    - 排名得分在 0-1 或 0-100 之间，替代原来硬性的 0 和 1。
    - 如果最大回撤突破 default_max_dd 强行赋予极低的分数惩罚。
    - 发现 future_max_return >= tp_target 时给予极大的褒奖分数。
    - 会自动生成 group 信息用于 lgb_ranker / xgb_ranker。
    """
    print(f"[Label V18] 启动 LTR 截面排序标签生成，强行要求 {tp_target} 爆发力目标。")
    
    if "future_max_return" not in df.columns:
        print("[Label V18] 警告: 未检测到未来连词数据，只可用于最新预测模式。")
        df = df.with_columns(
            pl.lit(0.0).alias("future_max_return"),
            pl.lit(0.0).alias("future_max_drawdown")
        )

    # 确定按天截面 Group
    group_col = "trade_date" if "trade_date" in df.columns else ("period_start" if "period_start" in df.columns else None)
    
    if group_col is not None:
        # [V18] 标签重构: 增强对极致回撤的断头台非线性二次方惩罚，同时添加暴涨的强烈奖励
        df = df.with_columns(
            pl.when(pl.col("future_max_return") >= tp_target)
            .then(pl.col("future_close_return") + 5.0)  # 狂暴褒奖
            .when(pl.col("future_max_drawdown") < -0.03)
            .then(pl.col("future_close_return") - (pl.col("future_max_drawdown") ** 2) * 100)
            .otherwise(pl.col("future_close_return") + pl.col("future_max_drawdown") * 1.5)
            .alias("risk_adj_return")
        )
        
        # [V12] LTR 核心：以当天截面 risk_adj_return 的相对排名分位数作为回归/排序目标 (0~10)
        df = df.with_columns(
            (pl.col("risk_adj_return").rank(method="min", descending=False).over(group_col) / 
             (pl.count("万得代码").over(group_col) + 1e-9) * 10.0).alias("raw_rank_score")
        )
    else:
        # 单文件预测且无时间标识时降级为绝对阈值拟合 (理论上不应发生)
        df = df.with_columns(
            pl.when(pl.col("future_max_return") >= tp_target)
            .then(pl.col("future_close_return") + 5.0)
            .when(pl.col("future_max_drawdown") < -0.03)
            .then(pl.col("future_close_return") - (pl.col("future_max_drawdown") ** 2) * 100)
            .otherwise(pl.col("future_close_return") + pl.col("future_max_drawdown") * 1.5)
            .alias("risk_adj_return")
        )
        df = df.with_columns(
            pl.when(pl.col("risk_adj_return") >= tp_target).then(10.0)
            .when(pl.col("risk_adj_return") > 0).then(5.0)
            .otherwise(0.0).alias("raw_rank_score")
        )
        
    # [V11] 熔断惩罚防线：无论排名多高，回撤破底线严惩 (直接赋予 0，让 Ranker 极度厌恶)
    df = df.with_columns(
        pl.when(pl.col("future_max_drawdown") <= default_max_dd)
        .then(pl.lit(0))
        .otherwise(pl.col("raw_rank_score").round(0).cast(pl.Int32))
        .alias("label")
    )
    
    # 附带生成给排序模型专用的 group 标识 (将 datetime 转为 integer string 或 int)
    if group_col is not None:
        # 如果是 datetime
        if df.schema[group_col] in [pl.Datetime, pl.Date]:
            df = df.with_columns(pl.col(group_col).dt.strftime("%Y%m%d").cast(pl.Int32).alias("group_id"))
        else:
            df = df.with_columns(pl.col(group_col).cast(pl.Int32).alias("group_id"))
    else:
        df = df.with_columns(pl.lit(1).alias("group_id"))
        
    return df

def generate_rolling_features(df_agg: pl.DataFrame) -> pl.DataFrame:
    """V3 时序滑动窗口特征 (支持单券或全量数据)"""
    # 确保有数据
    if df_agg.height == 0:
        return df_agg
        
    # 如果是全量数据，需要按股票分组；如果是单券，直接 sort 即可
    group_cols = ["万得代码"] if "万得代码" in df_agg.columns else []
    
    df_agg = df_agg.sort(["period_start"])
    
    rolling_exprs = [
        pl.col("ofi_sum").rolling_mean(window_size=10, min_periods=1).alias("ofi_sum_ma10"),
        pl.col("volatility_mean").rolling_max(window_size=10, min_periods=1).alias("vol_max_10"),
        pl.col("price_diff_mean").rolling_mean(window_size=10, min_periods=1).alias("price_diff_ma10")
    ]
    
    if "order_book_turnover" in df_agg.columns:
        rolling_exprs.extend([
            pl.col("order_book_turnover").rolling_mean(window_size=5, min_periods=1).alias("order_book_turnover_ts_mean_5"),
            pl.col("order_book_turnover").rolling_std(window_size=5, min_periods=1).fill_null(0.0).alias("order_book_turnover_ts_std_5")
        ])
    
    # 日内量比 (Intraday Volume Ratios) 与 量能加速度
    if "cum_volume" in df_agg.columns:
        vol_1m = pl.col("cum_volume").diff(1).fill_null(0.0)
        rolling_exprs.extend([
            # 5分钟量比: 当前1分钟成交量 / 过去5分钟平均每分钟成交量
            (vol_1m / (vol_1m.rolling_mean(window_size=5, min_periods=1) + 1e-9)).alias("volume_ratio_5m"),
            # 15分钟量比: 当前1分钟成交量 / 过去15分钟平均每分钟成交量
            (vol_1m / (vol_1m.rolling_mean(window_size=15, min_periods=1) + 1e-9)).alias("volume_ratio_15m"),
            # 量能加速度 (cum_volume 的二阶导数: 速度的变化率)
            vol_1m.diff(1).fill_null(0.0).alias("vol_acceleration")
        ])
    
    # 宏观因子 rolling (如果存在)
    if "market_ofi_sum" in df_agg.columns:
        rolling_exprs.append(
            pl.col("market_ofi_sum").rolling_mean(window_size=10, min_periods=1).alias("market_ofi_ma10")
        )
    
    if group_cols:
        df_agg = df_agg.with_columns([expr.over(group_cols) for expr in rolling_exprs])
    else:
        df_agg = df_agg.with_columns(rolling_exprs)
    
    # === [V10.2] 订单流高阶 Rolling 因子 ===
    # 1. OFI 自相关一致性 (Persistence): 连续 N 个时段 OFI 同号的比率
    if "ofi_sum" in df_agg.columns:
        ofi_sign = pl.col("ofi_sum").sign()
        ofi_sign_prev = pl.col("ofi_sum").shift(1).sign()
        same_sign = pl.when(ofi_sign == ofi_sign_prev).then(pl.lit(1.0)).otherwise(pl.lit(0.0))
        persistence_expr = same_sign.rolling_mean(window_size=10, min_periods=1).alias("ofi_persistence")
        # 2. OFI 加速度 (Acceleration): OFI 的一阶差分
        accel_expr = pl.col("ofi_sum").diff(1).fill_null(0.0).alias("ofi_acceleration")
        if group_cols:
            df_agg = df_agg.with_columns([persistence_expr.over(group_cols), accel_expr.over(group_cols)])
        else:
            df_agg = df_agg.with_columns([persistence_expr, accel_expr])
    
    # 3. 日内动量截断 (Intraday Momentum): 当前价格相对当日开盘价的涨幅
    if "last_price" in df_agg.columns:
        if group_cols:
            df_agg = df_agg.with_columns(
                ((pl.col("last_price") / (pl.col("last_price").first().over(group_cols) + 1e-9)) - 1.0).alias("intraday_momentum")
            )
        else:
            first_price = df_agg.select(pl.col("last_price").first()).item()
            df_agg = df_agg.with_columns(
                ((pl.col("last_price") / (first_price + 1e-9)) - 1.0).alias("intraday_momentum")
            )
        
    # [V17] 注入原生微观毒性复合因子，供树模型自动学习伪轴特征
    toxic_exprs = []
    if "vpin" in df_agg.columns and "intraday_momentum" in df_agg.columns:
        toxic_exprs.append((pl.col("vpin") / (pl.col("intraday_momentum").abs() + 1e-9)).alias("toxic_breakout_trap"))
    if "limit_up_magnet_effect" in df_agg.columns and "smc_fvg_bearish" in df_agg.columns:
        toxic_exprs.append((pl.col("limit_up_magnet_effect") * pl.col("smc_fvg_bearish")).alias("false_magnet_drop"))
    if toxic_exprs:
        df_agg = df_agg.with_columns(toxic_exprs)

    # ==========================================
    # [V9] 注入高阶 Alpha 因子在线重建引擎
    # ==========================================
    factor_json_path = os.path.join(MODEL_DIR, "discovered_factors.json")
    if os.path.exists(factor_json_path):
        try:
            import json
            with open(factor_json_path, "r", encoding="utf-8") as f:
                factor_meta = json.load(f)
            
            deps_dict = factor_meta.get("factor_dependencies", {})
            advanced_exprs = []
            
            for f_name, bases in deps_dict.items():
                # 判断基础字段是否在本 dataframe 中存在
                if not all(b in df_agg.columns for b in bases):
                    continue
                    
                # 【1】双变量算子
                if len(bases) == 2:
                    fa, fb = bases[0], bases[1]
                    if "_div_" in f_name:
                        advanced_exprs.append((pl.col(fa) / (pl.col(fb) + 1e-9)).cast(pl.Float32).alias(f_name))
                    elif "_sub_" in f_name:
                        advanced_exprs.append((pl.col(fa) - pl.col(fb)).cast(pl.Float32).alias(f_name))
                    elif "_mul_" in f_name:
                        advanced_exprs.append((pl.col(fa) * pl.col(fb)).cast(pl.Float32).alias(f_name))
                
                # 【2】单变量算子
                elif len(bases) == 1:
                    fa = bases[0]
                    if f_name.endswith("_log1p_abs"):
                        advanced_exprs.append(pl.col(fa).abs().log1p().cast(pl.Float32).alias(f_name))
                    elif f_name.endswith("_sign_sqrt"):
                        advanced_exprs.append((pl.col(fa).sign() * pl.col(fa).abs().sqrt()).cast(pl.Float32).alias(f_name))
                    elif f_name.endswith("_ts_std_5") and group_cols:
                        advanced_exprs.append(pl.col(fa).rolling_std(window_size=5).over(group_cols).cast(pl.Float32).alias(f_name))
                    elif f_name.endswith("_ts_mean_5") and group_cols:
                        advanced_exprs.append(pl.col(fa).rolling_mean(window_size=5).over(group_cols).cast(pl.Float32).alias(f_name))
                    elif f_name.endswith("_cs_zscore") and "period_start" in df_agg.columns:
                        advanced_exprs.append(((pl.col(fa) - pl.col(fa).mean()) / (pl.col(fa).std() + 1e-9)).over("period_start").cast(pl.Float32).alias(f_name))
                    elif f_name.endswith("_rank"):
                        if "period_start" in df_agg.columns:
                            advanced_exprs.append((pl.col(fa).rank() / pl.count("万得代码")).over("period_start").cast(pl.Float32).alias(f_name))
                        else:
                            advanced_exprs.append(pl.col(fa).rank().cast(pl.Float32).alias(f_name))
            
            if advanced_exprs:
                # 批量应用高阶复合指标
                df_agg = df_agg.with_columns(advanced_exprs)
                print(f"[V9 动态重建] 成功加载 {len(advanced_exprs)} 个高阶复合算子因子")
                
        except Exception as e:
            print(f"[V9 动态重建 ERROR] 加载 discovered_factors.json 失败: {e}")
            
    return df_agg

# ===================== 三模型训练 (Learning to Rank) =====================

def _train_lgb(X, y, groups):
    # [V11] 转换为 Pairwise LTR 排序任务，拟合截面排名
    split = int(len(X) * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    g_tr, g_val = groups[:split], groups[split:]
    
    # 根据 group_id 计算每个 group 的 size
    def get_group_sizes(g_array):
        _, counts = np.unique(g_array, return_counts=True)
        return counts
        
    train_data = lgb.Dataset(X_tr, label=y_tr, group=get_group_sizes(g_tr))
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, group=get_group_sizes(g_val))
    
    params = {
        'objective': 'lambdarank', 'metric': 'ndcg', 'boosting_type': 'gbdt',
        'learning_rate': 0.03, 'num_leaves': 15,
        'feature_fraction': 0.7, 'lambda_l2': 0.5,
        'eval_at': [3, 5],
        'verbose': -1, 'seed': 42
    }
    gbm = lgb.train(
        params, train_data, num_boost_round=800,
        valid_sets=[val_data], valid_names=['val'],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=0)]
    )
    best_iter = gbm.best_iteration if gbm.best_iteration else 800
    print(f"    [LGB Ranker] Best iteration: {best_iter}")
    _safe_save_model(gbm.save_model, LGB_PATH)
    
    del train_data, val_data
    import gc; gc.collect()
    
    return gbm

def _train_xgb(X, y, groups):
    # [V11] XGBoost Ranker打分
    def get_group_sizes(g_array):
        _, counts = np.unique(g_array, return_counts=True)
        return counts
        
    split = int(len(X) * 0.8)
    
    dtrain = xgb.DMatrix(X[:split], label=y[:split])
    dtrain.set_group(get_group_sizes(groups[:split]))
    
    dval = xgb.DMatrix(X[split:], label=y[split:])
    dval.set_group(get_group_sizes(groups[split:]))
    
    params = {
        'objective': 'rank:pairwise', 'eval_metric': 'ndcg@3',
        'max_depth': 4, 'learning_rate': 0.03,
        'subsample': 0.8, 'colsample_bytree': 0.7,
        'seed': 42, 'verbosity': 0
    }
    bst = xgb.train(
        params, dtrain, num_boost_round=800,
        evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False
    )
    print(f"    [XGB Ranker] Best iteration: {bst.best_iteration}")
    _safe_save_model(bst.save_model, XGB_PATH)
    
    del dtrain, dval
    import gc; gc.collect()
    
    return bst

def _train_catboost(X, y, groups):
    # [V11] CatBoost YetiRank 打分
    split = int(len(X) * 0.8)
    pool_tr = cb.Pool(X[:split], label=y[:split], group_id=groups[:split])
    pool_val = cb.Pool(X[split:], label=y[split:], group_id=groups[split:])
    
    params = {
        'iterations': 200,
        'learning_rate': 0.05, 'depth': 4,
        'loss_function': 'YetiRank', 'eval_metric': 'NDCG:top=3',
        'verbose': 0, 'random_seed': 42,
        'early_stopping_rounds': 20
    }
    model = cb.CatBoostRanker(**params)
    try:
        model.fit(pool_tr, eval_set=pool_val)
        print(f"    [CB Ranker] Best iteration: {model.best_iteration_}")
        _safe_save_model(model.save_model, CB_PATH)
    except cb.CatBoostError as e:
        print(f"    [CB Ranker] 训练异常跳过: {e}")
        try:
            import shutil
            if os.path.isdir(CB_PATH): shutil.rmtree(CB_PATH, ignore_errors=True)
            elif os.path.exists(CB_PATH): os.remove(CB_PATH)
        except:
            pass
        return None
    
    del pool_tr, pool_val
    import gc; gc.collect()
    
    return model

def train_ensemble_v3(df_train: pl.DataFrame, features: list, *, train_date: str = ""):
    """V12 三模型集成训练 (含 V4 深度学习特征与 V5 市场环境状态与 V12 Importance 生命周期追踪)"""
    # [V15] 主动剔除僵尸因子以减负增效
    dead_path = os.path.join(MODEL_DIR, "dead_factors.json")
    if os.path.exists(dead_path):
        try:
            import json as _json
            with open(dead_path, "r", encoding="utf-8") as f:
                dead_data = _json.load(f)
            dead_list = dead_data.get("dead_factors", [])
            original_count = len(features)
            features = [f for f in features if f not in dead_list]
            if len(features) < original_count:
                print(f"[V15] 成功过滤掉 {original_count - len(features)} 个僵尸因子，当前激活基础特征数: {len(features)}")
        except Exception:
            pass

    # [V15 Hotfix] 保存最终真正参与训练激活的特征集，供预测精准对齐
    active_path = os.path.join(MODEL_DIR, "active_features.json")
    try:
        import json as _json
        with open(active_path, "w", encoding="utf-8") as f:
            _json.dump({"features": features}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 强制依据特征全集（features）来补齐字段，绝不使用动态的 available 子集
    missing = [f for f in features if f not in df_train.columns]
    if missing:
        print(f"[Ensemble] 缺失特征 (补零): {missing}")
        df_train = df_train.with_columns([pl.lit(0.0).alias(f) for f in missing])
    
    # ------------------ V5: GMM 市场状态嵌入 ------------------
    if "market_regime" in features:
        print("[Ensemble V5] 计算 GMM 无监督市场状态聚类...")
        regime_detector = MarketRegimeDetector(n_components=3, window=10)
        regime_detector.fit(df_train)
        regimes = regime_detector.predict(df_train)
        df_train = df_train.with_columns(pl.Series("market_regime", regimes))
        joblib.dump(regime_detector, REGIME_PATH)
        print("[Ensemble V5] GMM 市场状态打标完成并固化。")
    # -------------------------------------------------------------
    
    # 提取截面特征 X, 标签 y 和 组别 group_id
    y = df_train.select("label").to_numpy().flatten()
    groups = df_train.select("group_id").to_numpy().flatten() if "group_id" in df_train.columns else np.ones_like(y, dtype=np.int32)
    X = df_train.select(features).to_numpy()
    
    # ------------------ V4 Transformer 预训练 ------------------
    if HAS_DL:
        from dl_pipeline import train_transformer, extract_transformer_embeddings
        import torch
        print("[Ensemble DL] 开始训练 Transformer 模型并提取时序嵌入...")
        try:
            dl_model = train_transformer(df_train, features, seq_len=5, epochs=3)
            if dl_model is not None:
                torch.save(dl_model.state_dict(), TRANSFORMER_PATH)
                embs = extract_transformer_embeddings(dl_model, df_train, features, seq_len=5)
                X = np.concatenate([X, embs], axis=1)
                print(f"[Ensemble DL] Transformer 训练完成，成功拼入 {embs.shape[1]} 维特征。")
            else:
                X = np.concatenate([X, np.zeros((len(X), 32))], axis=1)
        except Exception as e:
            print(f"[Ensemble DL] 训练异常，全零填充维系管道: {e}")
            X = np.concatenate([X, np.zeros((len(X), 32))], axis=1)
    # --------------------------------------------------------------------
    
    # [补丁 V5] 终极净身：清理所有的 INF / -INF / NaN，防止引擎崩溃
    # 策略: 填零而非丢弃整行，因为大量 V10.2/V11 因子在部分日期不存在被补零后
    #        经 Transformer embedding 混入 NaN，导致 95%+ 样本被误杀
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.where(np.isinf(y), 0.0, y)
    
    # 只丢弃 label 为 NaN 的行（极少数）
    mask_valid = ~np.isnan(y)
    X = X[mask_valid]
    y = y[mask_valid]
    groups = groups[mask_valid]
    
    # [V11 LTR 排序修正] 由于 XGB 和 LGBM Ranker 要求同一个 group 的样本必须相邻，强制按 groups 排序
    sort_idx = np.argsort(groups)
    X = X[sort_idx]
    y = y[sort_idx]
    groups = groups[sort_idx]
    
    # [V11 修复] LightGBM lambdarank 硬限制: 每个 query group 最多 10000 行
    MAX_GROUP_SIZE = 8000
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    oversized = group_counts > MAX_GROUP_SIZE
    if oversized.any():
        print(f"[Ensemble] 发现 {oversized.sum()} 个 group 超过 {MAX_GROUP_SIZE} 行限制，正在截断...")
        keep_mask = np.ones(len(X), dtype=bool)
        for gid, cnt in zip(unique_groups[oversized], group_counts[oversized]):
            g_indices = np.where(groups == gid)[0]
            drop_indices = np.random.default_rng(42).choice(g_indices, size=int(cnt - MAX_GROUP_SIZE), replace=False)
            keep_mask[drop_indices] = False
        X = X[keep_mask]
        y = y[keep_mask]
        groups = groups[keep_mask]
    
    # [性能优化] 训练集超过 MAX_TRAIN_ROWS 时随机采样，避免 CatBoost YetiRank 的 O(n²) 爆炸
    MAX_TRAIN_ROWS = 200000
    if len(X) > MAX_TRAIN_ROWS:
        print(f"[Ensemble] 训练集 {len(X)} 行超过上限 {MAX_TRAIN_ROWS}，随机采样...")
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(X), size=MAX_TRAIN_ROWS, replace=False)
        sample_idx.sort()  # 保持 group 排序
        X = X[sample_idx]
        y = y[sample_idx]
        groups = groups[sample_idx]
    
    print(f"[Ensemble] 数据清洗与 LTR 分组就绪，最终样本数/初始样本数: {len(X)}/{len(mask_valid)}")
    
    import gc
    
    print("[Ensemble] 训练 LTR LightGBM Ranker...")
    lgb_model = _train_lgb(X, y, groups)
    gc.collect()
    
    # [V8] Item 8: 输出 Top-20 特征重要性报告
    # [V12] 新增 Importance 历史追踪 + 僵尸因子检测
    import json as _json
    import csv as _csv
    try:
        importance = lgb_model.feature_importance(importance_type='gain')
        # 特征名: 如果 Transformer embedding 被 concat 了，维度可能超过 features
        num_raw = len(features)
        feat_names = list(features) + [f"dl_emb_{i}" for i in range(len(importance) - num_raw)]
        
        # 强制强类型解析，消除静态检查器的 Unknown 切片警示
        imp_pairs = []
        for fn, imp in zip(feat_names[:len(importance)], importance.tolist()):
            imp_pairs.append((str(fn), float(imp)))
        
        imp_pairs.sort(key=lambda x: -x[1])
        print("\n[Feature Importance] Top-20 因子贡献度:")
        
        for rank, (fname, score) in enumerate(imp_pairs[:20], 1):
            print(f"    {rank:2d}. {fname:40s} | Gain: {score:.1f}")
        imp_path = os.path.join(MODEL_DIR, "feature_importance.json")
        with open(imp_path, "w", encoding="utf-8") as f:
            _json.dump(dict(imp_pairs[:50]), f, ensure_ascii=False, indent=2)
        print(f"[Feature Importance] 已保存至 {imp_path}")
        
        # [V12] 追加写入 importance_history.csv (长表格式)
        history_path = os.path.join(MODEL_DIR, "importance_history.csv")
        is_new = not os.path.exists(history_path)
        date_tag = train_date if train_date else "unknown"
        with open(history_path, "a", newline="", encoding="utf-8") as hf:
            writer = _csv.writer(hf)
            if is_new:
                writer.writerow(["train_date", "factor_name", "gain"])
            for fname, score in imp_pairs:
                writer.writerow([date_tag, fname, f"{score:.4f}"])
        print(f"[Feature Importance V12] 历史追踪已追加至 {history_path}")
        
        # [V12] 僵尸因子检测 (Dead Factor Detection)
        _detect_dead_features(history_path)
        
    except Exception as e:
        print(f"[Feature Importance] 提取失败: {e}")
    
    print("[Ensemble] 训练 LTR XGBoost Ranker...")
    xgb_model = _train_xgb(X, y, groups)
    gc.collect()
    
    print("[Ensemble] 训练 LTR CatBoost Ranker...")
    cb_model = _train_catboost(X, y, groups)
    gc.collect()
    
    print(f"[Ensemble] 模型权重均已保存至 {MODEL_DIR}")
    return lgb_model, xgb_model, cb_model


def _detect_dead_features(history_path: str, lookback_rounds: int = 3) -> None:
    """检测连续 N 轮训练中 Gain 均为 0 的僵尸因子，写入 dead_factors.json"""
    import json as _json
    try:
        df_hist = pl.read_csv(history_path)
        if df_hist.height == 0:
            return
        
        # 获取最近 N 轮的 train_date
        recent_dates = (
            df_hist.select("train_date").unique()
            .sort("train_date", descending=True)
            .head(lookback_rounds)
            .to_series().to_list()
        )
        
        if len(recent_dates) < lookback_rounds:
            return  # 训练轮次不够，暂不检测
        
        # 筛选最近 N 轮数据
        df_recent = df_hist.filter(pl.col("train_date").is_in(recent_dates))
        
        # 按因子聚合: 所有轮次 Gain 均为 0 的就是僵尸因子
        df_dead = (
            df_recent.group_by("factor_name")
            .agg([
                pl.col("gain").cast(pl.Float64).max().alias("max_gain"),
                pl.col("gain").cast(pl.Float64).sum().alias("total_gain"),
                pl.len().alias("n_rounds"),
            ])
            .filter(
                (pl.col("max_gain") <= 0.0) & (pl.col("n_rounds") >= lookback_rounds)
            )
            .sort("factor_name")
        )
        
        dead_list = df_dead.select("factor_name").to_series().to_list()
        
        # [Fix] 白名单保护：防止核心基础因子被误杀
        WHITELIST = {"near_limit_up_accumulation", "smc_fvg_bearish", "smc_fvg_bullish", "cross_book_depletion_rate", "cross_trade_order_divergence"}
        dead_list = [f for f in dead_list if f not in WHITELIST]
        
        if dead_list:
            dead_path = os.path.join(MODEL_DIR, "dead_factors.json")
            with open(dead_path, "w", encoding="utf-8") as f:
                _json.dump({
                    "lookback_rounds": lookback_rounds,
                    "recent_dates": recent_dates,
                    "dead_factors": dead_list,
                    "count": len(dead_list),
                }, f, ensure_ascii=False, indent=2)
            print(f"\033[33m[V12 Dead Factor] 检测到 {len(dead_list)} 个僵尸因子 (连续 {lookback_rounds} 轮 Gain=0):\033[0m")
            for df_name in dead_list[:10]:
                print(f"    [*] DEAD: {df_name}")
            if len(dead_list) > 10:
                print(f"    ... 及其他 {len(dead_list) - 10} 个")
            print(f"    详情已保存至: {dead_path}")
        else:
            print(f"[V12 Dead Factor] 未检测到僵尸因子 (最近 {lookback_rounds} 轮)")
    except Exception as e:
        print(f"[V12 Dead Factor] 检测失败: {e}")


def predict_ensemble_top3(df_latest: pl.DataFrame, features: list):
    """V3 三模型投票选股 (含 V4 深度学习特征与 V5 市场环境状态)"""
    # [V15 Hotfix] 强制对齐训练时实际激活的特征集，保证模型输入 shape 正确
    active_path = os.path.join(MODEL_DIR, "active_features.json")
    if os.path.exists(active_path):
        try:
            import json as _json
            with open(active_path, "r", encoding="utf-8") as f:
                active_data = _json.load(f)
            features = active_data.get("features", features)
        except Exception:
            pass

    missing = [f for f in features if f not in df_latest.columns]
    if missing:
        df_latest = df_latest.with_columns([pl.lit(0.0).alias(f) for f in missing])
    
    # ------------------ V5: 预测集 GMM 市场状态注入 ------------------
    if "market_regime" in features and os.path.exists(REGIME_PATH):
        try:
            regime_detector = joblib.load(REGIME_PATH)
            regimes = regime_detector.predict(df_latest)
            df_latest = df_latest.with_columns(pl.Series("market_regime", regimes))
        except Exception as e:
            print(f"[Ensemble V5 DL] 无法加载 MarketRegime 模型，赋予默认震荡态 0: {e}")
            df_latest = df_latest.with_columns(pl.lit(0.0).alias("market_regime"))
    elif "market_regime" in features:
        df_latest = df_latest.with_columns(pl.lit(0.0).alias("market_regime"))
    # ------------------------------------------------------------------    
    
    X_np = df_latest.select(features).to_numpy()
    
    # ----------- 注入 Transformer 记忆 -----------
    if HAS_DL and os.path.exists(TRANSFORMER_PATH):
        try:
            dl_model = TemporalFeatureExtractor(input_dim=len(features), embed_dim=32, num_heads=4, num_layers=2)
            dl_model.load_state_dict(torch.load(TRANSFORMER_PATH, map_location="cpu", weights_only=True))
            embs = extract_transformer_embeddings(dl_model, df_latest, features, seq_len=5)
            X_np = np.concatenate([X_np, embs], axis=1)
        except Exception as e:
            print(f"[Ensemble DL] 加载 Transformer 侧模型时出错，使用全零特征: {e}")
            X_np = np.concatenate([X_np, np.zeros((len(X_np), 32))], axis=1)
    elif HAS_DL:
        # 兼容: 如果之前没有生成 transformer 模型但预测期望它的结构，补全默认零特征对齐架构
        X_np = np.concatenate([X_np, np.zeros((len(X_np), 32))], axis=1)
    # -----------------------------------------
    
    # [补丁] 同样要在预测时清洗预测数据的 INF 和 NaN，保持与训练阶段完全一致的逻辑以防止数据偏倚
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 使用临时文件安全加载模型，避开 C++ 组件无法解析含有中文路径的致命缺陷
    def load_lgb(path):
        return lgb.Booster(model_file=path)
    lgb_model = _safe_load_model(lgb, load_lgb, LGB_PATH)
    p_lgb = lgb_model.predict(X_np) if lgb_model else np.zeros(len(X_np))
    
    def load_xgb(path):
        bst = xgb.Booster()
        bst.load_model(path)
        return bst
    xgb_model = _safe_load_model(xgb, load_xgb, XGB_PATH)
    p_xgb = xgb_model.predict(xgb.DMatrix(X_np)) if xgb_model else np.zeros(len(X_np))
    
    def load_cb(path):
        mdl = cb.CatBoostRanker()
        mdl.load_model(path)
        return mdl
    cb_model = _safe_load_model(cb, load_cb, CB_PATH)
    p_cb = cb_model.predict(X_np) if cb_model else np.zeros(len(X_np))
    
    # [V11 LTR] 输出的已经是相对强度得分 (Score)，不需要转化为概率，直接按比例平均
    avg_score = (p_lgb + p_xgb + p_cb) / 3.0
    
    df_res = df_latest.with_columns([
        pl.Series("lgb_score", p_lgb),
        pl.Series("xgb_score", p_xgb),
        pl.Series("cb_score", p_cb),
        pl.Series("ensemble_avg", avg_score)
    ])
    
    # [V11 LTR 修复] 纯动态门槛: 取 95 分位数，不再有概率时代的 0.85 硬下限
    if df_res.height > 0:
        p_95 = df_res.select(pl.col("ensemble_avg").quantile(0.95)).item()
        vote_counts = ((avg_score >= p_95) & (avg_score > 0)).astype(int) * 3
        df_res = df_res.with_columns(pl.Series("vote_count", vote_counts))
    else:
        df_res = df_res.with_columns(pl.lit(0).alias("vote_count"))
        
    df_voted = df_res.filter(pl.col("vote_count") >= 2)
    
    # [Debug Output]
    if df_res.height > 0:
        max_score = df_res.select(pl.col("ensemble_avg").max()).item()
        print(f"    [Predict Debug] 全市场最高综合得分 (ensemble_avg): {max_score:.4f}  | 动态 95 分位门槛: {p_95:.4f}")
        if df_voted.height == 0:
            print("    [Predict Debug] 即使降低门槛也无标的达成，今日维持空仓看戏。")
    
    # [V16] 提取需要的拦截特征给状态机
    cols_to_select = [
        "万得代码", "ensemble_avg", "vote_count",
        "lgb_score", "xgb_score", "cb_score",
        "last_price", "ofi_sum", "max_limit_up_order_amt"
    ]
    if "big_net_inflow" in df_voted.columns:
        cols_to_select.append("big_net_inflow")
    if "vpin" in df_voted.columns:
        cols_to_select.append("vpin")

    top_3 = (
        df_voted.sort("ensemble_avg", descending=True)
        .head(3)
        .select(cols_to_select)
    )

    # [V17.0] 终极解禁: 移除导致严重误杀的 Z-Score 方差阈值，完全信任 LTR 树阵列的原生排位
    return top_3

