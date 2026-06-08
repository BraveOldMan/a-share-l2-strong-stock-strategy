"""
ML 自动因子生成引擎 (Alpha Mining Engine)
=========================================
暴力挖掘 → IC 筛选 → 正交去冗 → 自动注入

从现有基础因子两两组合，通过 4 种算子产生候选因子，
经 Rank IC 筛选和相关性过滤后，输出高价值正交因子集。
"""
import polars as pl
import numpy as np
import json
import os
from typing import Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DISCOVERED_PATH = os.path.join(_SCRIPT_DIR, "models", "discovered_factors.json")

def generate_cross_factors(
    df: pl.DataFrame,
    base_features: list[str],
    operators: Optional[list[str]] = None,
    target_names: Optional[list[str]] = None,
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """
    [V9 高级版] 对基础因子两两组合，并应用时序/截面等高级算子。
    Args:
        df: 输入 DataFrame（需包含 base_features, 万得代码, period_start/自然日）
        base_features: 基础因子列名列表
        operators: 要使用的算子列表
    Returns:
        (扩展后的 DataFrame, 新增的候选因子结构字典 {因子名: [依赖基础特征], ...})
    """
    if operators is None:
        operators = ["ratio", "diff", "product", "log1p_abs", "sign_sqrt", "rank", "ts_std_5", "ts_mean_5", "cs_zscore"]

    available = [f for f in base_features if f in df.columns]
    if len(available) < 2:
        print("[AlphaMine] 可用基础因子不足 2 个，跳过交叉生成。")
        return df, {}

    new_cols: list[pl.Expr] = []
    new_names: list[str] = []
    factor_deps: dict[str, list[str]] = {} # 记录生成因子的基础特征依赖，由于 V9 加入复杂算子，在预测时必须按相同法则重构

    # 如果有时间信息，确保先排好序供时序因子使用
    if "万得代码" in df.columns:
        if "period_start" in df.columns:
            df = df.sort(["万得代码", "period_start"])
            time_col = "period_start"
        elif "自然日" in df.columns:
            df = df.sort(["万得代码", "自然日"])
            time_col = "自然日"
        else:
            time_col = None
    else:
        time_col = None

    for i, fa in enumerate(available):
        # --- 1. 单变量复杂算子 (时序 / 截面) ---
        if "log1p_abs" in operators:
            name = f"{fa}_log1p_abs"
            if target_names is None or name in target_names:
                new_cols.append(pl.col(fa).abs().log1p().cast(pl.Float32).alias(name))
                new_names.append(name)
                factor_deps[name] = [fa]
            
        if "sign_sqrt" in operators:
            name = f"{fa}_sign_sqrt"
            if target_names is None or name in target_names:
                new_cols.append((pl.col(fa).sign() * pl.col(fa).abs().sqrt()).cast(pl.Float32).alias(name))
                new_names.append(name)
                factor_deps[name] = [fa]

        if "ts_std_5" in operators and time_col is not None:
            name = f"{fa}_ts_std_5"
            if target_names is None or name in target_names:
                new_cols.append(pl.col(fa).rolling_std(window_size=5).over("万得代码").cast(pl.Float32).alias(name))
                new_names.append(name)
                factor_deps[name] = [fa]

        if "ts_mean_5" in operators and time_col is not None:
            name = f"{fa}_ts_mean_5"
            if target_names is None or name in target_names:
                new_cols.append(pl.col(fa).rolling_mean(window_size=5).over("万得代码").cast(pl.Float32).alias(name))
                new_names.append(name)
                factor_deps[name] = [fa]

        # [V19 NEW] 指数移动平均 (EMA) - 对最近值赋予更高权重，平滑边界异常
        if "ewm_mean_5" in operators and time_col is not None:
            name = f"{fa}_ewm_mean_5"
            if target_names is None or name in target_names:
                new_cols.append(pl.col(fa).ewm_mean(com=5, ignore_nulls=True).over("万得代码").cast(pl.Float32).alias(name))
                new_names.append(name)
                factor_deps[name] = [fa]

        if "cs_zscore" in operators and time_col is not None:
            name = f"{fa}_cs_zscore"
            if target_names is None or name in target_names:
                new_cols.append(((pl.col(fa) - pl.col(fa).mean()) / (pl.col(fa).std() + 1e-9)).over(time_col).cast(pl.Float32).alias(name))
                new_names.append(name)
                factor_deps[name] = [fa]

        if "rank" in operators:
            name = f"{fa}_rank"
            if target_names is None or name in target_names:
                if time_col is not None:
                    new_cols.append((pl.col(fa).rank() / pl.count("万得代码")).over(time_col).cast(pl.Float32).alias(name))
                else:
                    new_cols.append(pl.col(fa).rank().cast(pl.Float32).alias(name))
                new_names.append(name)
                factor_deps[name] = [fa]

        # --- 2. 双变量算子 (交叉代数) ---
        for j, fb in enumerate(available):
            if i >= j: continue  # 跳过自身和重复对
            
            if "ratio" in operators:
                name = f"{fa}_div_{fb}"
                if target_names is None or name in target_names:
                    new_cols.append((pl.col(fa) / (pl.col(fb) + 1e-9)).cast(pl.Float32).alias(name))
                    new_names.append(name)
                    factor_deps[name] = [fa, fb]

            if "diff" in operators:
                name = f"{fa}_sub_{fb}"
                if target_names is None or name in target_names:
                    new_cols.append((pl.col(fa) - pl.col(fb)).cast(pl.Float32).alias(name))
                    new_names.append(name)
                    factor_deps[name] = [fa, fb]

            if "product" in operators:
                name = f"{fa}_mul_{fb}"
                if target_names is None or name in target_names:
                    new_cols.append((pl.col(fa) * pl.col(fb)).cast(pl.Float32).alias(name))
                    new_names.append(name)
                    factor_deps[name] = [fa, fb]

            # [V19 NEW] 对数差异：适合对量值差异极大的变量进行对比，平滑极值影响
            if "log_diff" in operators:
                name = f"{fa}_logdiff_{fb}"
                if target_names is None or name in target_names:
                    expr = (pl.col(fa).abs().log1p() - pl.col(fb).abs().log1p()).cast(pl.Float32)
                    new_cols.append(pl.when(expr.is_infinite() | expr.is_nan()).then(0.0).otherwise(expr).alias(name))
                    new_names.append(name)
                    factor_deps[name] = [fa, fb]

    # 内存优化：如果未指定 target_names，说明是大规模挖掘，采用评估后合并机制避免 OOM
    batch_size = 50
    final_cols = []
    final_names = []
    
    # 动态评估参数 (如果传递了的话，我们可以在外部或内部过滤，由于原签名没有 ic_eval_label，
    # 我们为了精简，直接在这里如果不带 target_names 则每次微批计算并清除不用的列)
    # 实际上，最好只在这里挂载表达式集合，交由外部处理。
    # 为了保护原逻辑，如果有 target_names，直接全部应用：
    if target_names is not None:
        for start in range(0, len(new_cols), batch_size):
            batch = new_cols[start : start + batch_size]
            df = df.with_columns(batch)
        print(f"[AlphaMine] 重建了 {len(new_names)} 个预定候选因子")
        return df, factor_deps

    # 以下为 "挖掘模式"，为避免 OOM，我们返回全量表达式列表，不直接挂载进 df
    # 由 run_alpha_mining 调用方去接管分批计算！
    return df, factor_deps, new_cols, new_names

# ===================== 第二阶段: IC 筛选 =====================

def screen_by_ic(
    df: pl.DataFrame,
    candidate_cols: list[str],
    label_col: str = "label",
    ic_threshold: float = 0.02,
) -> list[tuple[str, float]]:
    """
    [V8] 衰减加权 Rank IC 筛选:
    将数据按时间分为 3 段 (前1/3, 中1/3, 后1/3)，
    分别计算 IC，用 [0.2, 0.3, 0.5] 加权，最近时段占 50%。
    """
    if label_col not in df.columns:
        print(f"[AlphaMine] 标签列 '{label_col}' 不存在，跳过 IC 筛选。")
        return [(c, 0.0) for c in candidate_cols]

    label_arr = df.select(label_col).to_numpy().flatten().astype(np.float64)
    n = len(label_arr)
    
    # 三段切分点
    seg1, seg2 = n // 3, 2 * n // 3
    segments = [(0, seg1), (seg1, seg2), (seg2, n)]
    weights = [0.2, 0.3, 0.5]  # 最近时段权重最大
    
    kept: list[tuple[str, float]] = []

    from scipy.stats import rankdata

    for col in candidate_cols:
        if col not in df.columns:
            continue
        arr = df.select(col).to_numpy().flatten().astype(np.float64)

        weighted_ic = 0.0
        valid_segments = 0
        for (s, e), w in zip(segments, weights):
            if e - s < 30:
                continue
            seg_arr = arr[s:e]
            seg_label = label_arr[s:e]
            valid_mask = np.isfinite(seg_arr) & np.isfinite(seg_label)
            if valid_mask.sum() < 30:
                continue
            x_v = seg_arr[valid_mask]
            y_v = seg_label[valid_mask]
            if np.std(x_v) < 1e-12:
                continue
            rx = rankdata(x_v)
            ry = rankdata(y_v)
            ic = np.corrcoef(rx, ry)[0, 1]
            if np.isfinite(ic):
                weighted_ic += w * ic
                valid_segments += 1

        if valid_segments > 0 and abs(weighted_ic) >= ic_threshold:
            kept.append((col, float(weighted_ic)))

    kept.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"[AlphaMine V8] 衰减加权 IC 筛选: {len(candidate_cols)} 候选 → {len(kept)} 通过 (|IC| >= {ic_threshold})")
    return kept


# ===================== 第三阶段: 正交去冗 =====================

def filter_by_orthogonality(
    df: pl.DataFrame,
    ic_ranked: list[tuple[str, float]],
    corr_threshold: float = 0.7,
) -> list[tuple[str, float]]:
    """
    贪心正交过滤: 按 |IC| 从高到低遍历，与已纳入因子的相关系数超过阈值则剔除。

    Returns:
        过滤后的 (因子名, IC值) 列表
    """
    if not ic_ranked:
        return []

    accepted: list[tuple[str, float]] = []
    accepted_arrays: list[np.ndarray] = []

    for col, ic_val in ic_ranked:
        if col not in df.columns:
            continue
        arr = df.select(col).to_numpy().flatten().astype(np.float64)

        # 检查与所有已接受因子的相关性
        is_redundant = False
        for existing_arr in accepted_arrays:
            valid = np.isfinite(arr) & np.isfinite(existing_arr)
            if valid.sum() < 30:
                continue
            corr = abs(np.corrcoef(arr[valid], existing_arr[valid])[0, 1])
            if np.isfinite(corr) and corr >= corr_threshold:
                is_redundant = True
                break

        if not is_redundant:
            accepted.append((col, ic_val))
            accepted_arrays.append(arr)

    print(f"[AlphaMine] 正交过滤: {len(ic_ranked)} 候选 → {len(accepted)} 正交因子 (|corr| < {corr_threshold})")
    return accepted


# ===================== 第四阶段: 一键挖掘 =====================

def run_alpha_mining(
    df_train: pl.DataFrame,
    base_features: list[str],
    label_col: str = "label",
    ic_threshold: float = 0.02,
    corr_threshold: float = 0.7,
    max_new_factors: int = 20,
    max_rows_for_mining: int = 50000,
) -> list[str]:
    """
    一键执行全流程 Alpha 挖掘:
    1. 暴力组合生成候选因子 (包含高级 TS/CS 算子)
    2. IC 筛选
    3. 正交过滤
    4. 持久化结果

    Args:
        df_train: 训练数据 (必须含 label_col)
        base_features: 基础因子名列表
        label_col: 标签列名
        ic_threshold: IC 绝对值最低阈值
        corr_threshold: 正交过滤的相关系数上限
        max_new_factors: 最多保留的新因子数量
        max_rows_for_mining: IC 测算采样上限行数

    Returns:
        发现的新因子名列表
    """
    print("\n" + "=" * 60)
    print("【Alpha Mining Engine V9 启动】")
    print("=" * 60)

    # 内存优化: 当数据量过大时随机采样用于因子挖掘
    if df_train.height > max_rows_for_mining:
        print(f"[AlphaMine] 数据量 {df_train.height} 行超过阈值 {max_rows_for_mining}，随机采样...")
        df_mining = df_train.sample(n=max_rows_for_mining, seed=42)
    else:
        df_mining = df_train

    # [V15 优化] Shallow-LGBM 先验过滤防维数灾难
    max_base_features = 30
    if len(base_features) > max_base_features:
        print(f"[AlphaMine] 基础特征数量 {len(base_features)} > {max_base_features}，启动 Shallow-LGBM 先验筛选...")
        import lightgbm as lgb
        df_pd = df_mining.select(base_features + [label_col]).drop_nulls().to_pandas()
        
        # 保护机制: 如果 drop_nulls 把数据删光了，就回退
        if len(df_pd) > 100:
            X_tree = df_pd[base_features]
            y_tree = df_pd[label_col]

            tree_model = lgb.LGBMRegressor(n_estimators=30, num_leaves=15, random_state=42, n_jobs=-1, verbose=-1)
            tree_model.fit(X_tree, y_tree)

            imp = tree_model.feature_importances_
            sorted_feat = [f for _, f in sorted(zip(imp, base_features), reverse=True)]
            
            base_features = sorted_feat[:max_base_features]
            print(f"[AlphaMine] 筛选完毕！保留 Top {max_base_features} | Top 5: {base_features[:5]}")
            del df_pd, X_tree, y_tree, tree_model
        else:
            print("[AlphaMine] 警告: 去除空值后数据量过少，放弃 LGBM 筛选，默认取前 30 个。")
            base_features = base_features[:max_base_features]
        import gc; gc.collect()

    # 内存优化: 仅使用轻量算子，跳过需要 .over() 窗口的时序算子
    lightweight_ops = ["ratio", "diff", "product", "log1p_abs", "sign_sqrt", "rank"]

    # 阶段 1: 获取候选表达式 (阻止直接并入以防 OOM)
    res = generate_cross_factors(df_mining, base_features, operators=lightweight_ops)
    if len(res) == 2: # 异常或被当作 target_names 处理
        df_expanded, factor_deps = res
        candidates = list(factor_deps.keys())
        ic_ranked = screen_by_ic(df_expanded, candidates, label_col, ic_threshold)
    else:
        _, factor_deps, ext_exprs, ext_names = res
        candidates = []
        ic_ranked = []
        
        # 核心内存救星: 分批将列附着，测算 IC，抛弃废弃列
        batch_size = 50
        print(f"[AlphaMine] 发现 {len(ext_exprs)} 个潜在表达式，正在进行微批流式 IC 测试...")
        keep_exprs = []
        import gc
        for start in range(0, len(ext_exprs), batch_size):
            batch_exprs = ext_exprs[start : start + batch_size]
            batch_names = ext_names[start : start + batch_size]
            
            # 微批外挂
            df_batch = df_mining.with_columns(batch_exprs)
            
            # 使用现有 screen_by_ic 逻辑
            batch_ic_ranked = screen_by_ic(df_batch, batch_names, label_col, ic_threshold)
            ic_ranked.extend(batch_ic_ranked)
            
            # 保存及格选手的表达式
            passed_names = {x[0] for x in batch_ic_ranked}
            for expr, name in zip(batch_exprs, batch_names):
                if name in passed_names:
                    keep_exprs.append(expr)
                    candidates.append(name)
                    
            del df_batch
            gc.collect()

        print(f"[AlphaMine] 微批测算完毕，仅保留 {len(keep_exprs)} 个达标因子入内存。")
        # 仅将经过了生死考验的达标因子最终合并进 df_expanded 供后续正交过滤使用
        df_expanded = df_mining.with_columns(keep_exprs)

    del df_mining
    import gc; gc.collect()

    if not candidates or not ic_ranked:
        print("[AlphaMine] 没有因子通过 IC 筛选。")
        return []

    # 阶段 3: 正交过滤
    orthogonal = filter_by_orthogonality(df_expanded, ic_ranked, corr_threshold)

    # 截断到最大数量
    final = orthogonal[:max_new_factors]

    # 打印报告
    print(f"\n{'=' * 60}")
    print(f"【Alpha Mining 最终报告】")
    print(f"{'=' * 60}")
    print(f"  基础因子数:     {len(base_features)}")
    print(f"  候选因子数:     {len(candidates)}")
    print(f"  IC 筛选通过:    {len(ic_ranked)}")
    print(f"  正交过滤通过:   {len(orthogonal)}")
    print(f"  最终采纳:       {len(final)}")
    print(f"\n  {'Rank':>4}  {'Factor':40s}  {'IC':>8}")
    print(f"  {'-'*4}  {'-'*40}  {'-'*8}")
    for rank, (name, ic) in enumerate(final, 1):
        print(f"  {rank:4d}  {name:40s}  {ic:+.4f}")

    # 持久化
    discovered_names = [name for name, _ in final]
    final_deps = {name: factor_deps[name] for name in discovered_names}
    
    os.makedirs(os.path.dirname(DISCOVERED_PATH), exist_ok=True)
    with open(DISCOVERED_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "discovered_features": discovered_names,
            "ic_values": {name: ic for name, ic in final},
            "base_features_used": base_features,
            "factor_dependencies": final_deps
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  已持久化至: {DISCOVERED_PATH}")

    return discovered_names


def load_discovered_factors() -> list[str]:
    """加载已发现的因子列表"""
    if not os.path.exists(DISCOVERED_PATH):
        return []
    with open(DISCOVERED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("discovered_features", [])
