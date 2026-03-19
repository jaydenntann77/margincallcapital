"""
signals/momentum.py
────────────────────────────────────────────────────────────────
Cross-Sectional Momentum Signal

Strategy basis (from EDA):
  - 8h lookback momentum has the best cross-sectional Sharpe (1.94)
  - Universe avg AC(1) = -0.0072 → slight mean-reversion at lag-1,
    but positive momentum signal emerges at lag-2 through lag-6
  - Inverse-volatility weighting required: daily vol ranges from
    0.96% (TRX) to 9.48% (MIRA) — a 10x spread

Signal pipeline per hourly rebalance:
  1. Compute 8h momentum score per coin  (primary)
  2. Compute 4h momentum score           (secondary, blended)
  3. Compute rolling 14h ATR-based vol   (for inv-vol sizing)
  4. Apply volume filter                 (exclude illiquid at this bar)
  5. Rank → select top-N → size by inv-vol → normalize weights

Usage:
    from signals.momentum import MomentumSignal
    sig = MomentumSignal(con)
    weights = sig.compute(as_of_ts)   # → dict {symbol: weight}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass — all tunable params in one place
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MomentumConfig:
    # Signal lookback windows (in 1h bars)
    lookback_primary:   int   = 8      # EDA winner — best Sharpe 1.94
    lookback_secondary: int   = 4      # blended for smoothness
    weight_primary:     float = 0.70   # 70% 8h, 30% 4h
    weight_secondary:   float = 0.30

    # Universe / portfolio construction
    top_n:              int   = 10     # max long positions
    min_n:              int   = 3      # don't trade if fewer coins qualify
    max_weight:         float = 0.25   # max 25% in any single coin
    min_vol_usd_per_h:  float = 50_000 # min $50k hourly dollar volume

    # Volatility scaling
    vol_lookback:       int   = 14     # ATR lookback in hours
    vol_floor:          float = 0.002  # floor to avoid div-by-zero (0.2% per hour)
    vol_target:         float = 0.01   # target 1% hourly vol per position

    # Data requirements
    # Must be > lookback_primary so the momentum calc always has enough rows.
    # Keep it small — we fetch plenty of history anyway via hours_needed.
    min_bars_required:  int   = 10     # need ≥ 10 hourly bars to score a coin
    interval:           str   = '1m'   # raw data interval in DuckDB

    # Rebalance
    rebalance_freq_h:   int   = 1      # every 1 hour


# ─────────────────────────────────────────────────────────────────────────────
# Main signal class
# ─────────────────────────────────────────────────────────────────────────────

class MomentumSignal:
    """
    Computes cross-sectional momentum weights for all coins at a given timestamp.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Read-only connection to ohlcv_1m.duckdb
    universe : list[str]
        List of DuckDB symbol names, e.g. ['BTC-USD', 'ETH-USD', ...]
    cfg : MomentumConfig
        All tunable parameters
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        universe: Optional[List[str]] = None,
        cfg: Optional[MomentumConfig] = None,
    ):
        self.con  = con
        self.cfg  = cfg or MomentumConfig()
        self.universe = universe or _default_universe()
        logger.info(
            "MomentumSignal initialised | universe=%d coins | lookback=%dh/%dh | top_n=%d",
            len(self.universe), self.cfg.lookback_primary, self.cfg.lookback_secondary, self.cfg.top_n,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self, as_of: Optional[datetime] = None) -> Dict[str, float]:
        """
        Compute target portfolio weights as of `as_of` timestamp.

        Returns
        -------
        dict  {symbol: weight}  — weights sum to ≤ 1.0, all ≥ 0 (long-only)
        Returns empty dict {} if fewer than cfg.min_n coins qualify.
        """
        as_of = as_of or datetime.now(timezone.utc)
        as_of = as_of.replace(minute=0, second=0, microsecond=0)  # align to hour

        hourly = self._load_hourly_bars(as_of)
        if hourly.empty:
            logger.warning("No hourly bars loaded for %s — returning empty weights", as_of)
            return {}

        scores  = self._score_coins(hourly)
        weights = self._size_positions(scores, hourly)
        return weights

    def compute_batch(self, timestamps: List[datetime]) -> pd.DataFrame:
        """
        Compute weights for a list of timestamps (used in backtesting).

        Returns
        -------
        pd.DataFrame  index=timestamp, columns=symbols, values=weights
        """
        records = []
        for ts in timestamps:
            w = self.compute(ts)
            w['ts'] = ts
            records.append(w)
        df = pd.DataFrame(records).set_index('ts').fillna(0.0)
        return df

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_hourly_bars(self, as_of: datetime) -> pd.DataFrame:
        """
        Load and resample 1m → 1h bars.
        Fetches enough history to compute the longest lookback + vol window.
        """
        # Fetch enough bars to satisfy ALL downstream requirements:
        #   - min_bars_required (for dropna thresh filter)
        #   - lookback_primary  (for momentum calculation)
        #   - vol_lookback      (for ATR calculation)
        # Add a 10-bar safety buffer on top
        hours_needed = max(
            self.cfg.min_bars_required,
            self.cfg.lookback_primary,
            self.cfg.vol_lookback,
        ) + 10
        since = as_of - timedelta(hours=hours_needed)

        symbol_list = ", ".join(f"'{s}'" for s in self.universe)

        query = f"""
            SELECT
                DATE_TRUNC('hour', ts)              AS ts_hour,
                symbol,
                FIRST(open  ORDER BY ts)            AS open,
                MAX(high)                           AS high,
                MIN(low)                            AS low,
                LAST(close  ORDER BY ts)            AS close,
                SUM(volume)                         AS volume,
                COUNT(*)                            AS n_1m_bars
            FROM ohlcv
            WHERE
                interval = '{self.cfg.interval}'
                AND symbol IN ({symbol_list})
                AND ts >= '{since.strftime('%Y-%m-%d %H:%M:%S')}'
                AND ts <  '{as_of.strftime('%Y-%m-%d %H:%M:%S')}'
            GROUP BY DATE_TRUNC('hour', ts), symbol
            ORDER BY symbol, ts_hour
        """

        try:
            df = self.con.execute(query).df()
        except Exception as e:
            logger.error("DuckDB query failed: %s", e)
            return pd.DataFrame()

        if df.empty:
            return df

        df['ts_hour'] = pd.to_datetime(df['ts_hour'], utc=True)

        # Dollar volume filter: exclude bars where liquidity is too thin
        df['dollar_vol'] = df['close'] * df['volume']

        logger.debug("Loaded %d hourly bars for %d symbols", len(df), df['symbol'].nunique())
        return df

    # ── Signal computation ────────────────────────────────────────────────────

    def _score_coins(self, hourly: pd.DataFrame) -> pd.Series:
        """
        Compute composite momentum score for each coin.

        Score = weight_primary × rank(mom_primary)
              + weight_secondary × rank(mom_secondary)

        Both components are cross-sectionally ranked (0→1 percentile),
        then blended. This makes the signal robust to outliers.

        Returns
        -------
        pd.Series  index=symbol, values=composite_score (0→1)
        """
        pivot = (
            hourly
            .pivot_table(index='ts_hour', columns='symbol', values='close')
            .sort_index()
        )

        # Drop coins with insufficient history
        min_bars = self.cfg.min_bars_required
        pivot = pivot.dropna(axis=1, thresh=min_bars)

        if pivot.shape[1] < self.cfg.min_n:
            logger.warning("Only %d coins have sufficient history (need %d)", pivot.shape[1], self.cfg.min_n)
            return pd.Series(dtype=float)

        # ── Momentum scores (log return over lookback window) ──────────────
        def log_mom(df: pd.DataFrame, window: int) -> pd.Series:
            """Log return from `window` bars ago to last bar."""
            if len(df) < window + 1:
                return pd.Series(dtype=float)
            last  = df.iloc[-1]
            past  = df.iloc[-(window + 1)]
            mom   = np.log(last / past)
            return mom.dropna()

        mom_primary   = log_mom(pivot, self.cfg.lookback_primary)
        mom_secondary = log_mom(pivot, self.cfg.lookback_secondary)

        if mom_primary.empty:
            logger.warning("Not enough bars for primary momentum (need %d)", self.cfg.lookback_primary + 1)
            return pd.Series(dtype=float)

        # ── Cross-sectional rank (percentile 0→1) ─────────────────────────
        # Ranking neutralises the absolute level — only relative strength matters
        common = mom_primary.index.intersection(mom_secondary.index)
        if len(common) < self.cfg.min_n:
            return pd.Series(dtype=float)

        rank_primary   = mom_primary[common].rank(pct=True)
        rank_secondary = mom_secondary[common].rank(pct=True)

        composite = (
            self.cfg.weight_primary   * rank_primary +
            self.cfg.weight_secondary * rank_secondary
        )

        logger.debug(
            "Scores computed | coins=%d | top: %s",
            len(composite),
            composite.nlargest(3).index.tolist(),
        )
        return composite

    # ── Position sizing ───────────────────────────────────────────────────────

    def _size_positions(
        self,
        scores: pd.Series,
        hourly: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Convert composite scores → portfolio weights.

        Steps:
        1. Select top-N coins by composite score
        2. Apply hourly dollar-volume filter
        3. Compute inverse-ATR vol weight
        4. Scale weights to sum to 1.0
        5. Cap at max_weight, renormalise

        Returns
        -------
        dict  {symbol: weight}
        """
        if scores.empty or len(scores) < self.cfg.min_n:
            return {}

        # ── Step 1: Select top-N ───────────────────────────────────────────
        top_coins = scores.nlargest(self.cfg.top_n).index.tolist()

        # ── Step 2: Liquidity check at current bar ─────────────────────────
        last_bar = (
            hourly
            .groupby('symbol')
            .apply(lambda g: g.sort_values('ts_hour').iloc[-1])
            [['dollar_vol']]
        )
        liquid_coins = last_bar[
            last_bar['dollar_vol'] >= self.cfg.min_vol_usd_per_h
        ].index.tolist()

        selected = [c for c in top_coins if c in liquid_coins]

        if len(selected) < self.cfg.min_n:
            logger.warning(
                "Only %d liquid coins in top-%d after vol filter — skipping rebalance",
                len(selected), self.cfg.top_n,
            )
            return {}

        # ── Step 3: ATR-based inverse volatility weights ────────────────────
        # ATR over vol_lookback hours — measures recent realised volatility
        inv_vol_weights = {}
        for sym in selected:
            sym_df = hourly[hourly['symbol'] == sym].sort_values('ts_hour')
            if len(sym_df) < self.cfg.vol_lookback + 1:
                # Fallback: use std of recent close returns
                close = sym_df['close'].values
                if len(close) < 2:
                    inv_vol_weights[sym] = 1.0 / len(selected)
                    continue
                ret_std = np.std(np.diff(np.log(close))) + self.cfg.vol_floor
                inv_vol_weights[sym] = 1.0 / ret_std
                continue

            # True Range
            high  = sym_df['high'].values
            low   = sym_df['low'].values
            close = sym_df['close'].values
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]

            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - prev_close),
                    np.abs(low  - prev_close),
                )
            )
            atr = tr[-self.cfg.vol_lookback:].mean()

            # Normalise ATR by close price → ATR% (dimensionless)
            atr_pct = atr / close[-1]
            atr_pct = max(atr_pct, self.cfg.vol_floor)

            inv_vol_weights[sym] = 1.0 / atr_pct

        # ── Step 4: Normalise to sum = 1.0 ────────────────────────────────
        total = sum(inv_vol_weights.values())
        if total <= 0:
            # Equal weight fallback
            n = len(selected)
            return {sym: 1.0 / n for sym in selected}

        weights = {sym: v / total for sym, v in inv_vol_weights.items()}

        # ── Step 5: Cap at max_weight and renormalise ──────────────────────
        weights = _cap_and_renorm(weights, self.cfg.max_weight)

        logger.info(
            "Weights computed | n=%d | top: %s",
            len(weights),
            sorted(weights.items(), key=lambda x: -x[1])[:3],
        )
        return weights


# ─────────────────────────────────────────────────────────────────────────────
# Signal diagnostics — useful during research
# ─────────────────────────────────────────────────────────────────────────────

class SignalDiagnostics:
    """
    Wraps MomentumSignal to expose intermediate computations
    for inspection in Jupyter notebooks.
    """

    def __init__(self, signal: MomentumSignal):
        self.sig = signal

    def score_table(self, as_of: Optional[datetime] = None) -> pd.DataFrame:
        """
        Returns full score table for all coins at `as_of`.
        Includes: mom_8h, mom_4h, composite_score, rank, vol_atr_pct, inv_vol_weight.
        """
        as_of   = as_of or datetime.now(timezone.utc)
        as_of   = as_of.replace(minute=0, second=0, microsecond=0)
        hourly  = self.sig._load_hourly_bars(as_of)
        scores  = self.sig._score_coins(hourly)

        if scores.empty:
            return pd.DataFrame()

        pivot = (
            hourly
            .pivot_table(index='ts_hour', columns='symbol', values='close')
            .sort_index()
        )
        cfg = self.sig.cfg

        mom_8h = pd.Series({
            s: np.log(pivot[s].iloc[-1] / pivot[s].iloc[-(cfg.lookback_primary + 1)])
            for s in scores.index if s in pivot.columns and len(pivot[s].dropna()) > cfg.lookback_primary
        })
        mom_4h = pd.Series({
            s: np.log(pivot[s].iloc[-1] / pivot[s].iloc[-(cfg.lookback_secondary + 1)])
            for s in scores.index if s in pivot.columns and len(pivot[s].dropna()) > cfg.lookback_secondary
        })

        df = pd.DataFrame({
            'mom_8h':    mom_8h,
            'mom_4h':    mom_4h,
            'score':     scores,
        }).sort_values('score', ascending=False)

        df['rank']     = range(1, len(df) + 1)
        df['selected'] = df['rank'] <= cfg.top_n

        # ATR%
        atr_pcts = {}
        for sym in df.index:
            sym_df = hourly[hourly['symbol'] == sym].sort_values('ts_hour')
            if len(sym_df) < cfg.vol_lookback + 1:
                atr_pcts[sym] = np.nan
                continue
            high, low, close = sym_df['high'].values, sym_df['low'].values, sym_df['close'].values
            prev = np.roll(close, 1); prev[0] = close[0]
            tr   = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
            atr  = tr[-cfg.vol_lookback:].mean()
            atr_pcts[sym] = atr / close[-1] * 100

        df['atr_pct_14h'] = pd.Series(atr_pcts)

        weights = self.sig._size_positions(scores, hourly)
        df['weight'] = pd.Series(weights)
        df['weight'] = df['weight'].fillna(0.0)

        return df.round(6)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cap_and_renorm(weights: Dict[str, float], max_w: float) -> Dict[str, float]:
    """
    Iteratively cap weights at max_w and redistribute excess
    until all weights are within bounds.
    """
    w = dict(weights)
    for _ in range(20):  # max iterations
        capped   = {k: min(v, max_w) for k, v in w.items()}
        total    = sum(capped.values())
        if total <= 0:
            break
        w = {k: v / total for k, v in capped.items()}
        if all(v <= max_w + 1e-9 for v in w.values()):
            break
    return w


def _default_universe() -> List[str]:
    """Fallback universe — top 30 coins by liquidity from EDA."""
    return [
        'BTC-USD',  'ETH-USD',    'SOL-USD',    'XRP-USD',   'DOGE-USD',
        'BNB-USD',  'PAXG-USD',   'ZEC-USD',    'PEPE-USD',  'SUI-USD',
        'ADA-USD',  'TRX-USD',    'TAO-USD',    'LINK-USD',  'AVAX-USD',
        'NEAR-USD', 'TRUMP-USD',  'LTC-USD',    'ASTER-USD', 'WLFI-USD',
        'UNI-USD',  'PUMP-USD',   'DOT-USD',    'AAVE-USD',  'ENA-USD',
        'XPL-USD',  'FIL-USD',    'WLD-USD',    'VIRTUAL-USD','ICP-USD',
    ]


def load_universe_from_csv(path: str = '../data/tradeable_universe.csv') -> List[str]:
    """Load the tradeable universe saved by the EDA notebook."""
    try:
        return pd.read_csv(path)['symbol'].tolist()
    except FileNotFoundError:
        logger.warning("tradeable_universe.csv not found — using default top-30")
        return _default_universe()
