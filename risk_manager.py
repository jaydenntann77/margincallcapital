"""
Risk management layer.

Responsibilities
----------------
1. Validate weight-derived order quantities before placement.
2. Enforce per-asset maximum position size cap.
3. Track outstanding limit orders and surface stale ones for cancellation.
4. Track entry prices and trigger stop-losses.
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
        # Pending order tracking
        self._pending_orders: dict[str, dict]  = {}   # pair → order metadata
        # Stop-loss entry price tracking
        self._entry_prices:   dict[str, float] = {}   # pair → entry price

    # ------------------------------------------------------------------
    # Core order validation (weight-based)
    # ------------------------------------------------------------------

    def check_order(
        self,
        pair:     str,
        side:     Signal,
        quantity: float,
        price:    float,
    ) -> tuple[bool, str, float]:
        """Validate a weight-derived order quantity.

        Fetches a fresh portfolio snapshot so cash balances reflect any
        sells that already executed earlier in the same rebalance cycle.

        Returns (can_trade, reason, final_quantity).
        """
        portfolio = self._api.get_portfolio_value()
        total_usd = portfolio.get("total_usd", 0.0)

        if total_usd <= 0:
            return False, "Portfolio has no value", 0.0

        order_value = quantity * price

        if order_value < cfg.MIN_ORDER_VALUE_USD:
            return False, f"Order value too small (${order_value:.2f})", 0.0

        if side == Signal.BUY:
            usd_cash = portfolio.get("usd_cash", 0.0)

            if usd_cash < cfg.MIN_ORDER_VALUE_USD:
                return False, f"Insufficient USD cash (${usd_cash:.2f})", 0.0

            # Scale down to available cash if needed
            if order_value > usd_cash:
                quantity    = usd_cash / price
                order_value = usd_cash

            if order_value < cfg.MIN_ORDER_VALUE_USD:
                return False, f"Effective buy value too small (${order_value:.2f})", 0.0

            # Max-position cap
            asset       = pair.split("/")[0]
            current_val = portfolio.get("asset_values", {}).get(asset, {}).get("value_usd", 0.0)
            if (current_val + order_value) / total_usd > cfg.MAX_POSITION_FRAC:
                return False, f"{asset} would exceed max position ({cfg.MAX_POSITION_FRAC*100:.0f}%)", 0.0

            return True, "OK", quantity

        if side == Signal.SELL:
            asset         = pair.split("/")[0]
            asset_balance = portfolio.get("asset_values", {}).get(asset, {}).get("amount", 0.0)

            if asset_balance <= 0:
                return False, f"No {asset} holdings to sell", 0.0

            # Cap at actual holdings
            quantity    = min(quantity, asset_balance)
            order_value = quantity * price

            if order_value < cfg.MIN_ORDER_VALUE_USD:
                return False, f"Sell value too small (${order_value:.2f})", 0.0

            return True, "OK", quantity

        return False, f"Unsupported side: {side}", 0.0

    # ------------------------------------------------------------------
    # Order tracking
    # ------------------------------------------------------------------

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
        """Return [(pair, order_id), ...] for limit orders older than LIMIT_ORDER_TIMEOUT."""
        now   = time.time()
        stale = []
        for pair, info in self._pending_orders.items():
            age = now - info["placed_at"]
            if info["type"] == "LIMIT" and age > cfg.LIMIT_ORDER_TIMEOUT:
                stale.append((pair, info["order_id"]))
                logger.debug("Stale order: %s  id=%s  age=%.0fs", pair, info["order_id"], age)
        return stale

    def remove_order(self, pair: str) -> None:
        """Remove a tracked order (after cancellation or confirmed fill)."""
        self._pending_orders.pop(pair, None)

    # ------------------------------------------------------------------
    # Stop-loss tracking
    # ------------------------------------------------------------------

    def record_entry(self, pair: str, price: float) -> None:
        """Record the entry price after a successful BUY order."""
        self._entry_prices[pair] = price
        logger.info(
            "Entry recorded: %s @ %.5f (stop-loss at %.5f, -%.0f%%)",
            pair, price, price * (1 - cfg.STOP_LOSS_PCT), cfg.STOP_LOSS_PCT * 100,
        )

    def check_stop_loss(self, pair: str, current_price: float) -> bool:
        """Return True if price has dropped STOP_LOSS_PCT below the recorded entry."""
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
