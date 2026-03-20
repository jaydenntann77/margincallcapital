"""
EMA crossover signal generator.

Maintains per-pair EMA state incrementally (O(1) memory and CPU per update).

Signal logic
------------
  BUY  — fast EMA crosses ABOVE slow EMA  (bullish momentum)
  SELL — fast EMA crosses BELOW slow EMA  (bearish momentum)
  HOLD — no crossover this bar (or still in warmup)

Warmup
------
The strategy withholds signals for the first ``slow_period`` bars while the
two EMAs converge from their seed value (the first observed price) to
representative values.  Real signals only fire after that threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from logger import get_logger

logger = get_logger("Strategy")


class Signal(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class _PairState:
    """Mutable EMA state for a single trading pair."""
    fast_ema:      Optional[float] = None
    slow_ema:      Optional[float] = None
    prev_fast_ema: Optional[float] = None
    prev_slow_ema: Optional[float] = None
    bar_count:     int             = 0


class EMACrossoverStrategy:
    """
    Per-pair incremental EMA crossover strategy.

    Parameters
    ----------
    fast_period : int
        Short-term EMA period (default 9).
    slow_period : int
        Long-term EMA period (default 21).
    """

    def __init__(self, fast_period: int = 9, slow_period: int = 21) -> None:
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be strictly less than "
                f"slow_period ({slow_period})"
            )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._fast_alpha = 2.0 / (fast_period + 1)
        self._slow_alpha = 2.0 / (slow_period + 1)
        self._states: dict[str, _PairState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, pair: str, price: float) -> Signal:
        """
        Ingest a new price observation for *pair* and return the signal.

        Should be called exactly once per polling interval per pair.
        """
        state = self._states.setdefault(pair, _PairState())

        # Save previous values before overwriting (needed for crossover detection)
        state.prev_fast_ema = state.fast_ema
        state.prev_slow_ema = state.slow_ema

        # EMA update — seed with the very first price
        if state.fast_ema is None:
            state.fast_ema = price
            state.slow_ema = price
        else:
            state.fast_ema = (
                price * self._fast_alpha + state.fast_ema * (1 - self._fast_alpha)
            )
            state.slow_ema = (
                price * self._slow_alpha + state.slow_ema * (1 - self._slow_alpha)
            )

        state.bar_count += 1

        # Warmup guard — withhold signals until EMAs have had time to converge
        if state.bar_count < self.slow_period:
            logger.debug(
                "[%s] warmup %d/%d  price=%.5f  fast=%.5f  slow=%.5f",
                pair, state.bar_count, self.slow_period,
                price, state.fast_ema, state.slow_ema,
            )
            return Signal.HOLD

        # Need at least two observations to detect a direction change
        if state.prev_fast_ema is None or state.prev_slow_ema is None:
            return Signal.HOLD

        prev_fast_above = state.prev_fast_ema > state.prev_slow_ema
        curr_fast_above = state.fast_ema      > state.slow_ema

        if not prev_fast_above and curr_fast_above:
            sig = Signal.BUY
        elif prev_fast_above and not curr_fast_above:
            sig = Signal.SELL
        else:
            sig = Signal.HOLD

        logger.debug(
            "[%s] bar=%d  price=%.5f  fast=%.5f  slow=%.5f  spread=%+.5f  → %s",
            pair, state.bar_count, price,
            state.fast_ema, state.slow_ema,
            state.fast_ema - state.slow_ema,
            sig.value,
        )
        return sig

    def get_ema_values(self, pair: str) -> dict:
        """Return current EMA diagnostic values for *pair* (useful for logging)."""
        state = self._states.get(pair)
        if state is None:
            return {}
        spread = (
            state.fast_ema - state.slow_ema
            if state.fast_ema is not None and state.slow_ema is not None
            else None
        )
        return {
            "fast_ema":  state.fast_ema,
            "slow_ema":  state.slow_ema,
            "spread":    spread,
            "bar_count": state.bar_count,
            "warmed_up": state.bar_count >= self.slow_period,
        }

    def get_bar_count(self, pair: str) -> int:
        """Return the number of price bars seen for *pair* (0 if unseen)."""
        return self._states.get(pair, _PairState()).bar_count

    def reset(self, pair: Optional[str] = None) -> None:
        """Reset EMA state.  Pass pair=None to reset all pairs."""
        if pair is None:
            self._states.clear()
            logger.debug("Strategy state reset for all pairs.")
        else:
            self._states.pop(pair, None)
            logger.debug("Strategy state reset for %s.", pair)
