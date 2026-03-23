import pandas as pd
import numpy as np

class LongOnlyMomentumSignal:
    """Generate long-only cross-sectional momentum portfolio weights with vol sizing."""

    def __init__(self, 
                 fast_span: int = 5, 
                 slow_span: int = 20, 
                 vol_span: int = 250, 
                 z_score_threshold: float = 0.9, 
                 top_n: int = 3,
                 trend_filter_span: int = 200,
                 target_vol: float = 0.40,
                 annualization_factor: float = 365.0 * 6):
        """
        Args:
            fast_span: Span for the fast EMA.
            slow_span: Span for the slow EMA.
            vol_span: Span for the exponential standard deviation.
            z_score_threshold: Minimum z-score to be eligible.
            top_n: Max number of assets to hold.
            trend_filter_span: Span for absolute momentum exit signal.
            target_vol: Target annualized volatility for sizing (e.g., 0.40 for 40%).
            annualization_factor: 365 for daily crypto data, 24*365 for hourly, etc.
        """
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.vol_span = vol_span
        self.z_score_threshold = z_score_threshold
        self.top_n = top_n
        self.trend_filter_span = trend_filter_span
        self.target_vol = target_vol
        self.annualization_factor = annualization_factor

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        
        # 1. Calculate EMAs for momentum spread
        ma_s = prices.ewm(span=self.fast_span, adjust=False).mean()
        ma_l = prices.ewm(span=self.slow_span, adjust=False).mean()
        spread = ma_s - ma_l
        
        # 2. Spread Volatility normalization (Z-Score)
        spread_std = spread.ewm(span=self.vol_span, adjust=False).std()
        z_score = spread / spread_std.replace(0.0, np.nan)
        
        # 3. EXIT SIGNAL: Absolute Trend Filter
        # If price is below this MA, we do not want to hold it, even if relative momentum is high.
        trend_ma = prices.ewm(span=self.trend_filter_span, adjust=False).mean()
        uptrend_mask = prices > trend_ma
        
        # 4. Long-Only & Trend Filter
        # Z-score must be positive AND asset must be in an absolute uptrend
        valid_z = z_score.where((z_score > self.z_score_threshold) & uptrend_mask)
        
        # 5. Cross-Sectional Ranking
        ranks = valid_z.rank(axis=1, ascending=False, method='min')
        
        # Create boolean mask of selected assets
        is_selected = (ranks <= self.top_n)
        
        # 6. VOLATILITY SIZING
        # Calculate rolling annualized volatility of returns
        returns = prices.pct_change()
        asset_vol = returns.ewm(span=self.vol_span, adjust=False).std() * np.sqrt(self.annualization_factor)
        
        # Calculate raw inverse vol weights (target_vol / asset_vol)
        # Apply the 'is_selected' mask so non-selected assets get 0.0 weight
        raw_weights = (self.target_vol / asset_vol.replace(0.0, np.nan)) * is_selected.astype(float)
        
        # 7. Cash Buffer Management
        # If sum of weights > 1.0, scale down to 1.0.
        # If sum of weights < 1.0, we sit on the remaining cash.
        total_weight = raw_weights.sum(axis=1)
        
        # Scale factor limits total allocation to 100% max
        scale_factor = np.minimum(1.0, 1.0 / total_weight.replace(0.0, np.nan))
        
        # Final weights applied
        weights = raw_weights.mul(scale_factor, axis=0).fillna(0.0)
        
        return weights