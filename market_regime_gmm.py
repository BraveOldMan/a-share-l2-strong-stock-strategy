"""
A股市场状态识别 (Market Regime Detection) V5
基于 GMM 无监督聚类，提取 0 (震荡), 1 (趋势), 2 (高波极端)
严防未来函数：训练侧 fit，预测侧 transform
"""
import numpy as np
import polars as pl
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class MarketRegimeDetector:
    def __init__(self, n_components: int = 3, window: int = 10) -> None:
        self.n_components = n_components
        self.window = window # A股建议窗口小于币圈 (20->10)
        self.gmm = GaussianMixture(
            n_components=n_components,
            random_state=42,
            covariance_type="full",
        )
        self.scaler = StandardScaler()
        self._mapping: dict[int, int] = {}
        self._is_fitted: bool = False

    def _build_features(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """从 DataFrame 中提取状态特征 (vol, trend)。"""
        # Pipeline 中可能有 last_price 或 复权收盘价，优先使用 last_price
        price_col = "last_price" if "last_price" in df.columns else ("复权收盘价" if "复权收盘价" in df.columns else None)
        if price_col is None:
            return np.zeros((len(df), 2)), np.zeros(len(df), dtype=bool)
        prices = df.select(price_col).to_numpy().flatten().astype(np.float64)
        window = self.window

        returns = np.zeros_like(prices)
        returns[1:] = np.log(prices[1:] / prices[:-1])

        # 滚动波动率
        vol = np.zeros_like(prices)
        for i in range(window, len(prices)):
            vol[i] = np.std(returns[i - window : i])

        # 趋势强度
        trend = np.zeros_like(prices)
        for i in range(window, len(prices)):
            if prices[i - window] > 0:
                trend[i] = abs(prices[i] - prices[i - window]) / prices[i - window]

        X = np.column_stack([vol, trend])
        valid_mask = (vol > 0) & (trend > 0) & np.isfinite(vol) & np.isfinite(trend)
        return X, valid_mask

    def fit(self, df: pl.DataFrame) -> "MarketRegimeDetector":
        """仅用训练集拟合 GMM (防泄漏)。"""
        print(f"[MarketRegime] 启动 GMM fit (n={self.n_components})...")
        X, valid_mask = self._build_features(df)

        if valid_mask.sum() < 100:
            print("[MarketRegime] 警告: 数据量过少，无法进行有意义的聚类，跳过 fit。")
            return self

        X_train = X[valid_mask]
        X_scaled = self.scaler.fit_transform(X_train)
        self.gmm.fit(X_scaled)

        # 排序映射: Vol 最小 → 0, 中心 → 1, 最大 → 2
        regimes = self.gmm.predict(X_scaled)
        cluster_vols = []
        for i in range(self.n_components):
            mean_vol = X_train[regimes == i, 0].mean()
            cluster_vols.append((i, mean_vol))
        cluster_vols.sort(key=lambda x: x[1])
        
        self._mapping = {
            old_id: new_id for new_id, (old_id, _) in enumerate(cluster_vols)
        }
        self._is_fitted = True

        dist = np.bincount(
            np.array([self._mapping.get(r, 0) for r in regimes]),
            minlength=self.n_components,
        )
        print(f"[MarketRegime] 状态分布 (训练集): {dist}")
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """预测市场状态 (可用于测试集, 无泄漏)。"""
        X, valid_mask = self._build_features(df)
        final_regimes = np.zeros(len(df))

        if not self._is_fitted or valid_mask.sum() < 10:
            return final_regimes

        X_scaled = self.scaler.transform(X[valid_mask])
        raw_regimes = self.gmm.predict(X_scaled)
        final_regimes[valid_mask] = [self._mapping.get(r, 0) for r in raw_regimes]
        return final_regimes
