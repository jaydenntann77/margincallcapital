import pandas as pd
import numpy as np

class LongOnlyMomentumSignal:
    """Generate long-only cross-sectional momentum portfolio weights.

    This signal uses an EMA-spread z-score for each asset, filters for positive
    momentum, ranks assets cross-sectionally at each timestamp, and allocates
    equal weight to the top-ranked names.

    Signal computation summary:
        1. Compute fast and slow exponential moving averages (EMAs).
        2. Compute spread = EMA_fast - EMA_slow.
        3. Normalize spread by its exponential moving standard deviation to get
           a z-score.
        4. Keep assets with z-score above ``z_score_threshold``.
        5. Rank remaining assets by z-score (highest momentum first).
        6. Allocate equal weight across the top ``top_n`` assets.

    If fewer than ``top_n`` assets pass the threshold, unallocated capital
    remains as cash (weights sum to less than 1.0).
    """

    def __init__(self, fast_span: int = 24, slow_span: int = 72, vol_span: int = 260, z_score_threshold: float = 0.0, top_n: int = 20):
        """Initialize momentum signal parameters.

        Args:
            fast_span: Span for the fast EMA.
            slow_span: Span for the slow EMA.
            vol_span: Span for the exponential standard deviation used to
                normalize spread into z-scores.
            z_score_threshold: Minimum z-score required for an asset to be
                eligible for long allocation.
            top_n: Number of highest-ranked eligible assets to hold at each
                timestamp.
        """
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.vol_span = vol_span
        self.z_score_threshold = z_score_threshold
        self.top_n = top_n

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute long-only target weights from a price matrix.

        Args:
            prices: Wide price DataFrame where rows are timestamps and columns
                are asset symbols. Values should be close prices.

        Returns:
            pandas.DataFrame with the same shape as ``prices`` containing target
            portfolio weights per timestamp and symbol.

        Notes:
            - This function creates target weights, not executed weights.
              A backtester should usually apply a one-bar lag when executing.
            - Weights are equal-weighted among selected assets and can sum to
              less than 1.0 when few assets pass the threshold.
        """
        # 1. Calculate EMAs across all columns
        ma_s = prices.ewm(span=self.fast_span, adjust=False).mean()
        ma_l = prices.ewm(span=self.slow_span, adjust=False).mean()
        
        # 2. Calculate the Moving Average Spread
        spread = ma_s - ma_l
        
        # 3. Volatility normalization - Calculate the rolling standard deviation of the spread
        spread_std = spread.ewm(span=self.vol_span, adjust=False).std()
        z_score = spread / spread_std.replace(0.0, np.nan)
        
        # 4. Long-Only Filter: Ignore assets with negative momentum
        positive_z = z_score.where(z_score > self.z_score_threshold)
        
        # 5. Cross-Sectional Ranking
        # Rank the valid positive z-scores (ascending=False makes the highest momentum rank 1.0)
        ranks = positive_z.rank(axis=1, ascending=False, method='min')
        
        # 6. Weight Allocation (Equal weight to Top N) # TODO: Add option for volatility-adjusted weighting
        # Keep only the ones that are ranked <= top_n
        top_picks = ranks.where(ranks <= self.top_n, 0.0)
        
        # Set selected assets to 1.0, others to 0.0
        allocations = top_picks.where(top_picks == 0.0, 1.0)
        
        # Divide by the number of selected assets in that period to ensure weights sum to <= 1.0
        selected_count = allocations.sum(axis=1)
        weights = allocations.div(selected_count.replace(0.0, np.nan), axis=0).fillna(0.0)
        
        return weights