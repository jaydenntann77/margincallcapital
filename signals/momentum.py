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
    # EDA + sensitivity sweep winner: 6h (Sharpe 2.39, fwd_ret 0.020%)
    # vs original 8h (Sharpe 1.94). 3h secondary replaces 4h proportionally.
    lookback_primary:   int   = 6      # primary momentum window
    lookback_secondary: int   = 3      # secondary blend window
    weight_primary:     float = 0.70   # 70% 6h, 30% 3h
    weight_secondary:   float = 0.30

    # Universe / portfolio construction
    top_n:               int   = 8       # max long positions (tighter — higher conviction)
    min_n:               int   = 3       # don't trade if fewer coins qualify
    max_weight:          float = 0.25    # max 25% in any single coin
    min_vol_usd_per_h:   float = 50_000  # min $50k hourly dollar volume

    # Score quality gate — only hold coins scoring above this threshold.
    # Prevents low-momentum coins (TRX, PAXG) dominating via inv-vol sizing.
    # 0.60 = top ~40% of universe must have genuine positive momentum.
    min_score_threshold: float = 0.60

    # Volatility scaling
    vol_lookback:        int   = 14      # ATR lookback in hours
    vol_floor:           float = 0.002   # floor to avoid div-by-zero
    vol_target:          float = 0.01    # target 1% hourly vol per position

    # Data requirements
    min_bars_required:   int   = 10      # need >= 10 hourly bars to score a coin
    interval:            str   = '1m'    # raw data interval in DuckDB

    # Rebalance frequency and turnover gate.
    # Root cause: 38.95% mean turnover/hour = 0.935%/day fee drag.
    # Fix 1: rebalance every 4h instead of 1h (6x/day vs 24x/day).
    # Fix 2: skip the rebalance entirely if total |delta_weight| < 5%
    #         (avoids paying fees for noise-driven micro-shifts).
    rebalance_freq_h:       int   = 4    # recompute signal every 4 hours
    min_turnover_to_trade:  float = 0.05 # skip rebalance if total |Dw| < 5%


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

    def compute(
        self,
        as_of: Optional[datetime] = None,
        prev_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute target portfolio weights as of `as_of` timestamp.

        Parameters
        ----------
        as_of : datetime, optional
            Timestamp to compute signal for. Defaults to now().
        prev_weights : dict, optional
            Current portfolio weights {symbol: weight}.
            If provided, applies the turnover gate: if total |delta_weight|
            < cfg.min_turnover_to_trade, returns prev_weights unchanged
            (no trade, no fee).

        Returns
        -------
        dict  {symbol: weight}  — weights sum to <= 1.0, all >= 0 (long-only)
        Returns empty dict {} if fewer than cfg.min_n coins qualify.
        """
        as_of = as_of or datetime.now(timezone.utc)
        as_of = as_of.replace(minute=0, second=0, microsecond=0)  # align to hour

        hourly = self._load_hourly_bars(as_of)
        if hourly.empty:
            logger.warning("No hourly bars loaded for %s — returning empty weights", as_of)
            return prev_weights or {}

        scores, mom_raw = self._score_coins(hourly)
        weights = self._size_positions(scores, mom_raw, hourly)

        # ── Turnover gate ──────────────────────────────────────────────────
        # If new weights are too similar to current holdings, skip the trade.
        # Saves fees on noise-driven micro-shifts between rebalances.
        if prev_weights and weights:
            all_syms = set(weights) | set(prev_weights)
            turnover = sum(
                abs(weights.get(s, 0.0) - prev_weights.get(s, 0.0))
                for s in all_syms
            ) / 2
            if turnover < self.cfg.min_turnover_to_trade:
                logger.debug(
                    "Turnover gate: %.2f%% < %.2f%% threshold — holding current weights",
                    turnover * 100, self.cfg.min_turnover_to_trade * 100,
                )
                return prev_weights
        return weights

    def compute_batch(self, timestamps: List[datetime]) -> pd.DataFrame:
        """
        Compute weights for a list of timestamps (used in backtesting).

        Respects rebalance_freq_h: only calls compute() every N hours.
        Between rebalances, carries forward the last computed weights.
        Also passes prev_weights to each compute() call for the turnover gate.

        Returns
        -------
        pd.DataFrame  index=timestamp, columns=symbols, values=weights
        """
        records      = []
        prev_weights: Dict[str, float] = {}
        last_rebal_ts: Optional[datetime] = None

        for ts in timestamps:
            # Only recompute every rebalance_freq_h hours
            should_rebalance = (
                last_rebal_ts is None or
                (ts - last_rebal_ts).total_seconds() / 3600 >= self.cfg.rebalance_freq_h
            )

            if should_rebalance:
                new_weights  = self.compute(ts, prev_weights=prev_weights)
                if new_weights:
                    prev_weights  = new_weights
                    last_rebal_ts = ts

            w       = dict(prev_weights)
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
            return pd.Series(dtype=float), pd.Series(dtype=float)

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
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # ── Cross-sectional rank (percentile 0→1) ─────────────────────────
        # Ranking neutralises the absolute level — only relative strength matters.
        # We also return raw mom_primary so _size_positions can apply an
        # absolute momentum filter (coin must be going up, not just less down).
        common = mom_primary.index.intersection(mom_secondary.index)
        if len(common) < self.cfg.min_n:
            return pd.Series(dtype=float), pd.Series(dtype=float)

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
        return composite, mom_primary[common]

    # ── Position sizing ───────────────────────────────────────────────────────

    def _size_positions(
        self,
        scores: pd.Series,
        mom_raw: pd.Series,
        hourly: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Convert composite scores → portfolio weights.

        Steps:
        1a. Score quality gate  (relative rank >= min_score_threshold)
        1b. Absolute momentum gate  (raw 6h return must be > 0)
        1c. Select top-N from survivors
        2.  Liquidity check at current bar
        3.  Inverse-ATR vol weight
        4.  Normalise to sum = 1.0
        5.  Cap at max_weight, renormalise
        """
        if scores.empty or len(scores) < self.cfg.min_n:
            return {}

        # ── Step 1a: Relative score gate ───────────────────────────────────
        # Must be in top ~40% of universe by momentum rank.
        qualified = scores[scores >= self.cfg.min_score_threshold]

        # ── Step 1b: Absolute momentum gate ───────────────────────────────
        # Raw 6h log return must be strictly positive.
        # This is the key fix for TRX/PAXG dominance: they only enter
        # when they are genuinely trending up in absolute terms, not
        # just less-negative than a falling market.
        positive_mom = mom_raw[mom_raw > 0].index
        qualified    = qualified[qualified.index.isin(positive_mom)]

        if len(qualified) < self.cfg.min_n:
            logger.debug(
                "Only %d coins pass both gates (score>=%.2f AND mom>0) — holding cash",
                len(qualified), self.cfg.min_score_threshold,
            )
            return {}

        # ── Step 1c: Select top-N from qualified ──────────────────────────
        top_coins = qualified.nlargest(self.cfg.top_n).index.tolist()

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
        Includes: mom_primary, mom_secondary, composite_score, rank,
                  abs_mom_pass, score_pass, atr_pct_14h, weight, selected.
        """
        as_of  = as_of or datetime.now(timezone.utc)
        as_of  = as_of.replace(minute=0, second=0, microsecond=0)
        hourly = self.sig._load_hourly_bars(as_of)
        scores, mom_raw = self.sig._score_coins(hourly)

        if scores.empty:
            return pd.DataFrame()

        pivot = (
            hourly
            .pivot_table(index='ts_hour', columns='symbol', values='close')
            .sort_index()
        )
        cfg = self.sig.cfg

        mom_primary = pd.Series({
            s: np.log(pivot[s].iloc[-1] / pivot[s].iloc[-(cfg.lookback_primary + 1)])
            for s in scores.index if s in pivot.columns and len(pivot[s].dropna()) > cfg.lookback_primary
        })
        mom_secondary = pd.Series({
            s: np.log(pivot[s].iloc[-1] / pivot[s].iloc[-(cfg.lookback_secondary + 1)])
            for s in scores.index if s in pivot.columns and len(pivot[s].dropna()) > cfg.lookback_secondary
        })

        df = pd.DataFrame({
            f'mom_{cfg.lookback_primary}h':   mom_primary,
            f'mom_{cfg.lookback_secondary}h': mom_secondary,
            'score':                          scores,
        }).sort_values('score', ascending=False)

        df['rank']          = range(1, len(df) + 1)
        df['score_pass']    = df['score'] >= cfg.min_score_threshold
        df['abs_mom_pass']  = mom_raw.reindex(df.index) > 0
        df['both_pass']     = df['score_pass'] & df['abs_mom_pass']
        df['selected']      = False
        passing_top_n = df[df['both_pass']].head(cfg.top_n).index
        df.loc[passing_top_n, 'selected'] = True

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

        weights = self.sig._size_positions(scores, mom_raw, hourly)
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