"""
Risk management layer.

Responsibilities
----------------
1. Verify that sufficient balance is available before any order is placed.
2. Enforce a per-asset maximum position size cap.
3. Prevent duplicate orders through a per-pair signal cooldown window.
4. Track outstanding limit orders and surface stale ones for cancellation.
"""
from __future__ import annotations

import time
from typing import Optional, TYPE_CHECKING

from logger import get_logger
from strategy import Signal
import config as cfg

if TYPE_CHECKING:
    from client import RoostooClient

logger = get_logger("RiskManager")


class RiskManager:
    """Stateful risk-guardrail layer."""

    def __init__(self, api_client: "RoostooClient") -> None:
        self._api = api_client
        # Cooldown tracking
        self._last_signal_time: dict[str, float]  = {}   # pair → epoch seconds
        self._last_signal_type: dict[str, Signal] = {}   # pair → Signal
        # Pending order tracking
        self._pending_orders:   dict[str, dict]   = {}   # pair → order metadata
        # Stop-loss entry price tracking
        self._entry_prices:     dict[str, float]  = {}   # pair → entry price

    # ------------------------------------------------------------------
    # Core check
    # ------------------------------------------------------------------

    def check(
        self,
        pair:   str,
        signal: Signal,
        price:  float,
        share:  float = 1.0,  # New: percentage of total equity this strategy can use
    ) -> tuple[bool, str, float]:
        """Determine whether a signal passes risk checks based on a capital share."""
        # 1. Cooldown check (unchanged)
        if self._in_cooldown(pair, signal):
            remaining = int(cfg.SIGNAL_COOLDOWN_SECONDS - (time.time() - self._last_signal_time.get(pair, 0)))
            return False, f"Cooldown — {signal.value} within {remaining}s", 0.0

        # 2. Fetch live portfolio and scale by share
        portfolio = self._api.get_portfolio_value()
        full_total_usd = portfolio.get("total_usd", 0.0)
        
        # The sub-strategy only "sees" its portion of the total equity
        strat_total_usd = full_total_usd * share
        if strat_total_usd <= 0:
            return False, "Strategy allocation has no value", 0.0

        trade_value = strat_total_usd * cfg.TRADE_FRACTION

        # 3. Signal-specific checks
        if signal == Signal.BUY:
            # Scale available cash by the strategy share
            strat_usd_cash = portfolio.get("usd_cash", 0.0) * share
            
            if strat_usd_cash < cfg.MIN_ORDER_VALUE_USD:
                return False, f"Insufficient strategy cash (${strat_usd_cash:.2f})", 0.0

            # Max-position cap relative to the TOTAL portfolio for safety
            asset = pair.split("/")[0]
            current_val = portfolio.get("asset_values", {}).get(asset, {}).get("value_usd", 0.0)
            
            # We ensure this specific strategy's buy doesn't push the asset over the global limit
            if full_total_usd > 0 and (current_val + trade_value) / full_total_usd > cfg.MAX_POSITION_FRAC:
                return False, f"{asset} would exceed global max position cap", 0.0

            actual_value = min(trade_value, strat_usd_cash)
            quantity = actual_value / price
            return True, "OK", quantity

        if signal == Signal.SELL:
            asset = pair.split("/")[0]
            # Sell fraction applies to the total holdings of that asset
            asset_balance = portfolio.get("asset_values", {}).get(asset, {}).get("amount", 0.0)
            if asset_balance <= 0:
                return False, f"No {asset} to sell", 0.0

            quantity = asset_balance * cfg.SELL_FRACTION
            if (quantity * price) < cfg.MIN_ORDER_VALUE_USD:
                return False, "Sell value too small", 0.0
                
            return True, "OK", quantity

        return False, f"Unsupported signal: {signal}", 0.0

    # ------------------------------------------------------------------
    # Private per-direction checks
    # ------------------------------------------------------------------

    def _check_buy(
        self,
        pair:        str,
        price:       float,
        trade_value: float,
        portfolio:   dict,
        total_usd:   float,
    ) -> tuple[bool, str, float]:
        usd_cash = portfolio.get("usd_cash", 0.0)

        if usd_cash < cfg.MIN_ORDER_VALUE_USD:
            return False, f"Insufficient USD cash (${usd_cash:.2f})", 0.0

        # Max-position cap
        asset       = pair.split("/")[0]
        current_val = (
            portfolio.get("asset_values", {}).get(asset, {}).get("value_usd", 0.0)
        )
        if total_usd > 0 and (current_val + trade_value) / total_usd > cfg.MAX_POSITION_FRAC:
            return (
                False,
                f"{asset} would exceed max position cap "
                f"({cfg.MAX_POSITION_FRAC * 100:.0f}% of portfolio)",
                0.0,
            )

        # Cap trade to available cash
        actual_value = min(trade_value, usd_cash)
        if actual_value < cfg.MIN_ORDER_VALUE_USD:
            return False, f"Effective trade value too small (${actual_value:.2f})", 0.0

        quantity = actual_value / price
        return True, "OK", quantity

    def _check_sell(
        self,
        pair:      str,
        price:     float,
        portfolio: dict,
    ) -> tuple[bool, str, float]:
        asset         = pair.split("/")[0]
        # Read from the portfolio snapshot already fetched in check() — avoids a second API call
        asset_balance = portfolio.get("asset_values", {}).get(asset, {}).get("amount", 0.0)

        if asset_balance <= 0:
            return False, f"No {asset} holdings to sell", 0.0

        quantity   = asset_balance * cfg.SELL_FRACTION
        sell_value = quantity * price

        if sell_value < cfg.MIN_ORDER_VALUE_USD:
            return (
                False,
                f"Sell value too small (${sell_value:.2f} < min ${cfg.MIN_ORDER_VALUE_USD})",
                0.0,
            )

        return True, "OK", quantity

    def _in_cooldown(self, pair: str, signal: Signal) -> bool:
        """Return True if the same signal direction was acted on within the cooldown window."""
        if self._last_signal_type.get(pair) != signal:
            return False
        elapsed = time.time() - self._last_signal_time.get(pair, 0)
        return elapsed < cfg.SIGNAL_COOLDOWN_SECONDS

    # ------------------------------------------------------------------
    # State mutation — called by the bot after successful execution
    # ------------------------------------------------------------------

    def record_signal(self, pair: str, signal: Signal) -> None:
        """Record that we successfully acted on a signal (starts the cooldown timer)."""
        self._last_signal_time[pair] = time.time()
        self._last_signal_type[pair] = signal
        logger.debug("Signal recorded: %s → %s", pair, signal.value)

    def track_order(
        self,
        pair:       str,
        order_id:   str,
        order_type: str = "LIMIT",
        placed_at:  Optional[float] = None,
    ) -> None:
        """Register a newly placed order for stale-order tracking."""
        self._pending_orders[pair] = {
            "order_id":  order_id,
            "type":      order_type,
            "placed_at": placed_at or time.time(),
        }
        logger.debug("Tracking %s order %s for %s", order_type, order_id, pair)

    def get_stale_orders(self) -> list[tuple[str, str]]:
        """
        Return [(pair, order_id), ...] for limit orders that have not filled
        within LIMIT_ORDER_TIMEOUT seconds.
        """
        now   = time.time()
        stale = []
        for pair, info in self._pending_orders.items():
            age = now - info["placed_at"]
            if info["type"] == "LIMIT" and age > cfg.LIMIT_ORDER_TIMEOUT:
                stale.append((pair, info["order_id"]))
                logger.debug(
                    "Stale order detected: %s  id=%s  age=%.0fs",
                    pair, info["order_id"], age,
                )
        return stale

    def remove_order(self, pair: str) -> None:
        """Remove a tracked order (after cancellation or confirmed fill)."""
        self._pending_orders.pop(pair, None)

    # ------------------------------------------------------------------
    # Stop-loss tracking
    # ------------------------------------------------------------------

    def record_entry(self, pair: str, price: float) -> None:
        """Record the entry price after a successful BUY order fills."""
        self._entry_prices[pair] = price
        logger.info("Entry recorded: %s @ %.5f (stop-loss at %.5f, -%.0f%%)",
                    pair, price, price * (1 - cfg.STOP_LOSS_PCT), cfg.STOP_LOSS_PCT * 100)

    def check_stop_loss(self, pair: str, current_price: float) -> bool:
        """
        Return True if current price has dropped STOP_LOSS_PCT below entry.
        Only fires if we have a recorded entry price for this pair.
        """
        entry = self._entry_prices.get(pair)
        if entry is None:
            return False
        loss_pct = (current_price - entry) / entry
        if loss_pct <= -cfg.STOP_LOSS_PCT:
            logger.warning(
                "[STOP-LOSS] %s triggered — entry=%.5f  current=%.5f  loss=%.2f%%",
                pair, entry, current_price, loss_pct * 100,
            )
            return True
        return False

    def clear_entry(self, pair: str) -> None:
        """Clear a pair's entry price after the position is fully closed."""
        if pair in self._entry_prices:
            self._entry_prices.pop(pair)
            logger.debug("Entry price cleared for %s", pair)
