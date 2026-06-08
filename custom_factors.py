import numpy as np
from numba import njit, prange

@njit(parallel=True, nogil=True)
def factor_obi_momentum_ashare(total_bid_vols, total_ask_vols, window=20):
    """
    [V19 NEW] OBI 动量加速度 (OBI Momentum):
    衡量最近 window 个 tick (约1分钟) 的 OBI 一阶和二阶导数
    高阶反映挂单意愿加速度的翻脸翻脸速度
    """
    n_rows = total_bid_vols.shape[0]
    res_obi_mom = np.zeros(n_rows, dtype=np.float32)
    half_window = window // 2
    
    for i in prange(window, n_rows):
        b_curr, a_curr = total_bid_vols[i], total_ask_vols[i]
        obi_curr = (b_curr - a_curr) / (b_curr + a_curr) if (b_curr + a_curr) > 0 else 0.0
        
        b_half, a_half = total_bid_vols[i - half_window], total_ask_vols[i - half_window]
        obi_half = (b_half - a_half) / (b_half + a_half) if (b_half + a_half) > 0 else 0.0
        
        b_prev, a_prev = total_bid_vols[i - window], total_ask_vols[i - window]
        obi_prev = (b_prev - a_prev) / (b_prev + a_prev) if (b_prev + a_prev) > 0 else 0.0
        
        v1 = obi_curr - obi_half
        v2 = obi_half - obi_prev
        
        accel = v1 - v2
        
        res_obi_mom[i] = v1 + accel * 1.5
        
    return res_obi_mom

@njit(parallel=True, nogil=True)
def factor_liquidity_void_probe(bid_vols, ask_vols, levels=10):
    """
    [V19 NEW] 流动性真空探测针 (Liquidity Void Probe):
    对比前 3 档和后 7 档的挂单密度差。
    正值表示前档(容易被扫掉)的挂单厚度极大值大于后档极度匮乏，一触即溃即飞。
    """
    n_rows = bid_vols.shape[0]
    res_void = np.zeros(n_rows, dtype=np.float32)
    
    for i in prange(n_rows):
        b_front = 0.0
        b_back = 0.0
        a_front = 0.0
        a_back = 0.0
        
        for k in range(3):
            b_front += bid_vols[i, k]
            a_front += ask_vols[i, k]
            
        for k in range(3, levels):
            b_back += bid_vols[i, k]
            a_back += ask_vols[i, k]
            
        b_diff = (b_front / 3.0) - (b_back / 7.0)
        a_diff = (a_front / 3.0) - (a_back / 7.0)
        
        res_void[i] = b_diff - a_diff 

    return res_void



@njit(parallel=True, nogil=True)
def factor_ofi_multi_level_ashare(bid_prices, bid_vols, ask_prices, ask_vols, levels=10):
    n_rows = bid_prices.shape[0]
    res_multi_ofi = np.zeros(n_rows, dtype=np.float32)
    weights = np.linspace(1.0, 0.1, levels)
    
    for i in prange(1, n_rows):
        total_ofi = 0.0
        for k in range(levels):
            bp_t, bp_tm1 = bid_prices[i, k], bid_prices[i-1, k]
            bv_t, bv_tm1 = bid_vols[i, k], bid_vols[i-1, k]
            
            ap_t, ap_tm1 = ask_prices[i, k], ask_prices[i-1, k]
            av_t, av_tm1 = ask_vols[i, k], ask_vols[i-1, k]
            
            b_flow = 0.0
            if bp_t > bp_tm1:
                b_flow = bv_t
            elif bp_t < bp_tm1:
                b_flow = -bv_tm1
            else:
                b_flow = bv_t - bv_tm1
                
            a_flow = 0.0
            if ap_t > ap_tm1:
                a_flow = -av_tm1
            elif ap_t < ap_tm1:
                a_flow = av_t
            else:
                a_flow = av_tm1 - av_t
            total_ofi += (b_flow + a_flow) * weights[k]
        res_multi_ofi[i] = total_ofi
    return res_multi_ofi

@njit(parallel=True, nogil=True)
def factor_volatility_ashare(mid_prices, window=20):
    n_rows = mid_prices.shape[0]
    res_volatility = np.zeros(n_rows, dtype=np.float32)
    for i in prange(window, n_rows):
        sum_ret = 0.0
        sum_sq_ret = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            p_curr = mid_prices[j]
            p_prev = mid_prices[j-1]
            if p_prev > 0 and p_curr > 0:
                ret = np.log(p_curr / p_prev)
                sum_ret += ret
                sum_sq_ret += ret * ret
                count += 1
        if count > 1:
            mean_ret = sum_ret / count
            var_ret = (sum_sq_ret / count) - (mean_ret * mean_ret)
            if var_ret > 0:
                res_volatility[i] = np.sqrt(var_ret) * 10000 
    return res_volatility

@njit(parallel=True, nogil=True)
def factor_simple_obi_ashare(total_bid_vols, total_ask_vols):
    n_rows = total_bid_vols.shape[0]
    res_obi = np.zeros(n_rows, dtype=np.float32)
    for i in prange(n_rows):
        b_tot = total_bid_vols[i]
        a_tot = total_ask_vols[i]
        denom = b_tot + a_tot
        if denom > 0:
            res_obi[i] = (b_tot - a_tot) / denom
    return res_obi

# ================= V2 新增深度因子 =================

@njit(parallel=True, nogil=True)
def factor_weighted_price_diff_ashare(bid_prices, bid_vols, ask_prices, ask_vols, levels=5):
    """
    加权均价偏离度 (深度倾斜)：
    反映盘口挂单的重心是在压盘还是在托盘。
    """
    n_rows = bid_prices.shape[0]
    res_diff = np.zeros(n_rows, dtype=np.float32)
    
    for i in prange(n_rows):
        b_sum_val = 0.0
        b_sum_vol = 0.0
        a_sum_val = 0.0
        a_sum_vol = 0.0
        
        for k in range(levels):
            b_sum_val += bid_prices[i, k] * bid_vols[i, k]
            b_sum_vol += bid_vols[i, k]
            
            a_sum_val += ask_prices[i, k] * ask_vols[i, k]
            a_sum_vol += ask_vols[i, k]
            
        b_vwap = b_sum_val / b_sum_vol if b_sum_vol > 0 else 0.0
        a_vwap = a_sum_val / a_sum_vol if a_sum_vol > 0 else 0.0
        
        mid_price = (bid_prices[i, 0] + ask_prices[i, 0]) / 2.0
        
        # 挂单重心向上移动(托盘) vs 向下移动(压盘)
        b_skew = b_vwap - mid_price if mid_price > 0 else 0
        a_skew = a_vwap - mid_price if mid_price > 0 else 0
        
        res_diff[i] = b_skew + a_skew
        
    return res_diff

@njit(parallel=True, nogil=True)
def factor_depth_variance(bid_vols, ask_vols, levels=10):
    """
    盘口厚度方差：
    主力列阵时，某几笔单子会特别大，方差极大；散户混战时，各档位比较均匀，方差极小。
    """
    n_rows = bid_vols.shape[0]
    res_var = np.zeros(n_rows, dtype=np.float32)
    
    for i in prange(n_rows):
        total_vol = 0.0
        for k in range(levels):
            total_vol += bid_vols[i, k]
            total_vol += ask_vols[i, k]
            
        mean_vol = total_vol / (levels * 2)
        var_vol = 0.0
        for k in range(levels):
            var_vol += (bid_vols[i, k] - mean_vol) ** 2
            var_vol += (ask_vols[i, k] - mean_vol) ** 2
        
        res_var[i] = np.sqrt(var_vol / (levels * 2)) if mean_vol > 0 else 0
        
    return res_var


@njit(parallel=True, nogil=True)
def factor_book_slope(bid_vols, ask_vols, levels=10):
    """
    盘口弹性斜率 (Order Book Slope):
    对买卖合并挂单量 vs 档位序号做 OLS 线性回归。
    - 斜率 > 0: 深层挂单量更大 → 盘口纵深充裕，抗冲击能力强
    - 斜率 < 0: 前排挂单量更大 → 主力刺刀在前线，激进建仓/出货
    """
    n_rows = bid_vols.shape[0]
    res_slope = np.zeros(n_rows, dtype=np.float32)

    # 预计算 x 轴 (档位序号 0..levels-1) 的统计量
    x_mean = (levels - 1) / 2.0
    x_var_sum = 0.0
    for k in range(levels):
        x_var_sum += (k - x_mean) ** 2

    for i in prange(n_rows):
        # 合并买卖双边每档位的总挂单量
        y_sum = 0.0
        for k in range(levels):
            y_sum += bid_vols[i, k] + ask_vols[i, k]

        y_mean = y_sum / levels if levels > 0 else 0.0

        # OLS 斜率 = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
        cov_xy = 0.0
        for k in range(levels):
            vol = bid_vols[i, k] + ask_vols[i, k]
            cov_xy += (k - x_mean) * (vol - y_mean)

        if x_var_sum > 0:
            res_slope[i] = cov_xy / x_var_sum

    return res_slope

@njit(parallel=True, nogil=True)
def factor_depth_weighted_impact_ashare(mat_ap, mat_aa):
    """
    深度加权冲击成本 (Depth Weighted Price Impact)
    模拟吃掉卖盘全部 10 档后的加权均价 vs 买一价的偏离
    """
    n_rows = mat_ap.shape[0]
    depth_impact = np.zeros(n_rows, dtype=np.float32)
    for row_i in prange(n_rows):
        cum_vol = 0.0
        cum_amt = 0.0
        for lvl in range(10):
            p, v = mat_ap[row_i, lvl], mat_aa[row_i, lvl]
            if p > 0 and v > 0:
                cum_vol += v
                cum_amt += p * v
        if cum_vol > 0 and mat_ap[row_i, 0] > 0:
            depth_impact[row_i] = (cum_amt / cum_vol - mat_ap[row_i, 0]) / (mat_ap[row_i, 0] + 1e-9)
    return depth_impact

