"""
High-level API client for the Roostoo trading bot.

Wraps the low-level RoostooClient (client.py) and adds:
  - Retry with exponential backoff on transient network failures
  - cancel_order endpoint (POST /v3/cancel_order)
  - Convenience methods: prices, balances, portfolio valuation
  - Proper quantity / price formatting using per-pair exchange precision
"""
from __future__ import annotations

import math
import random
import time
from typing import Optional

# Initialise logging BEFORE importing client.py so that our root-logger
# configuration wins over client.py's logging.basicConfig() call.
from logger import get_logger

from client import RoostooClient
import config as cfg

logger = get_logger("ApiClient")


class BotApiClient:
    """Thread-safe (single-threaded) wrapper around RoostooClient."""

    def __init__(self) -> None:
        self._client = RoostooClient()
        self._exchange_cache: dict = {}   # populated lazily, never expires during a run

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call(self, method, *args, **kwargs) -> dict:
        """
        Call a RoostooClient method and retry on transient network failures.

        Network errors are detected by inspecting the error message inside the
        dict that client.py returns (it never raises — all exceptions are
        converted to error dicts by the underlying _request method).
        """
        last: dict = {}
        for attempt in range(cfg.MAX_RETRIES + 1):
            result = method(*args, **kwargs)
            last = result

            if isinstance(result, dict) and result.get("Success") is False:
                err = str(result.get("error", "")).lower()
                is_network = any(
                    kw in err
                    for kw in ("connection", "timeout", "network", "max retries", "refused", "reset by peer")
                )
                if is_network and attempt < cfg.MAX_RETRIES:
                    wait = cfg.RETRY_BACKOFF * (2 ** attempt) * (
                        1 + random.uniform(0, cfg.RETRY_JITTER)
                    )
                    logger.warning(
                        "Network error on attempt %d/%d: %s — retrying in %.1fs",
                        attempt + 1, cfg.MAX_RETRIES + 1, err, wait,
                    )
                    time.sleep(wait)
                    continue

            # Non-network error or success → return immediately
            return result

        logger.error("All %d retry attempts exhausted.", cfg.MAX_RETRIES + 1)
        return last

    # ------------------------------------------------------------------
    # Exchange metadata
    # ------------------------------------------------------------------

    def get_exchange_info(self) -> dict:
        """Fetch (and cache for the lifetime of this run) exchange metadata."""
        if not self._exchange_cache:
            result = self._call(self._client.get_exchange_info)
            if result.get("IsRunning") is not None:
                self._exchange_cache = result
            else:
                logger.warning("Unexpected exchange_info format: %s", result)
        return self._exchange_cache

    def get_trade_pairs(self) -> dict:
        """Return the TradePairs dict keyed by pair symbol."""
        return self.get_exchange_info().get("TradePairs", {})

    def get_pair_meta(self, pair: str) -> dict:
        """Return the metadata dict for a single pair (precision, MiniOrder, etc.)."""
        return self.get_trade_pairs().get(pair, {})

    def get_precision(self, pair: str) -> tuple[int, int]:
        """Return (price_precision, amount_precision) for a pair."""
        meta = self.get_pair_meta(pair)
        return meta.get("PricePrecision", 8), meta.get("AmountPrecision", 8)

    def validate_pair(self, pair: str) -> bool:
        """Return True if the pair exists on the exchange and CanTrade=True."""
        meta = self.get_pair_meta(pair)
        return bool(meta) and bool(meta.get("CanTrade", False))

    def _fmt_quantity(self, pair: str, quantity: float) -> str:
        """
        Format a quantity string respecting exchange AmountPrecision.

        AmountPrecision=0 means integer quantities (e.g. DOGE/USD, SHIB/USD).
        Truncates (floors) rather than rounds to avoid over-ordering.
        """
        _, amt_prec = self.get_precision(pair)
        if amt_prec == 0:
            return str(int(math.floor(quantity)))
        factor = 10 ** amt_prec
        truncated = math.floor(quantity * factor) / factor
        return f"{truncated:.{amt_prec}f}"

    def _fmt_price(self, pair: str, price: float) -> str:
        """Format a price string respecting exchange PricePrecision."""
        price_prec, _ = self.get_precision(pair)
        return f"{price:.{price_prec}f}"

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_price(self, pair: str) -> Optional[float]:
        """Return the last trade price for *pair*, or None on failure."""
        try:
            ticker = self._call(self._client.get_ticker, pair)
            if not ticker.get("Success"):
                logger.debug(
                    "Ticker Success=False for %s: %s", pair, ticker.get("ErrMsg")
                )
                return None
            pair_data = ticker.get("Data", {}).get(pair, {})
            raw = pair_data.get("LastPrice")
            return float(raw) if raw is not None else None
        except Exception as exc:
            logger.error("get_price(%s) raised: %s", pair, exc)
            return None

    # ------------------------------------------------------------------
    # Account / portfolio
    # ------------------------------------------------------------------

    def get_usd_balance(self) -> float:
        """Return the free USD balance."""
        try:
            df = self._client.get_balance_df()
            if df.empty:
                return 0.0
            row = df[df["asset"] == "USD"]
            return float(row["free"].iloc[0]) if not row.empty else 0.0
        except Exception as exc:
            logger.error("get_usd_balance failed: %s", exc)
            return 0.0

    def get_asset_balance(self, asset: str) -> float:
        """Return the free balance for *asset* (e.g. 'BTC')."""
        try:
            df = self._client.get_balance_df()
            if df.empty:
                return 0.0
            row = df[df["asset"] == asset]
            return float(row["free"].iloc[0]) if not row.empty else 0.0
        except Exception as exc:
            logger.error("get_asset_balance(%s) failed: %s", asset, exc)
            return 0.0

    def get_portfolio_value(self) -> dict:
        """
        Return a portfolio snapshot.

        Keys
        ----
        usd_cash     : float  — free USD
        asset_values : dict   — {asset: {amount, price, value_usd}}
        total_usd    : float  — estimated total value in USD
        """
        try:
            df = self._client.get_balance_df()
            if df.empty:
                return {"usd_cash": 0.0, "asset_values": {}, "total_usd": 0.0}

            usd_row  = df[df["asset"] == "USD"]
            usd_cash = float(usd_row["free"].iloc[0]) if not usd_row.empty else 0.0

            asset_values: dict = {}
            total_usd = usd_cash

            for _, row in df[df["asset"] != "USD"].iterrows():
                asset        = str(row["asset"])
                total_amount = float(row["total"])
                if total_amount <= 0:
                    continue
                price = self.get_price(f"{asset}/USD")
                if price:
                    value = total_amount * price
                    asset_values[asset] = {
                        "amount":    total_amount,
                        "price":     price,
                        "value_usd": value,
                    }
                    total_usd += value

            return {"usd_cash": usd_cash, "asset_values": asset_values, "total_usd": total_usd}

        except Exception as exc:
            logger.error("get_portfolio_value failed: %s", exc)
            return {"usd_cash": 0.0, "asset_values": {}, "total_usd": 0.0}

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_limit_buy(self, pair: str, quantity: float, price: float) -> dict:
        """Place a limit BUY order with exchange-correct precision."""
        qty_str   = self._fmt_quantity(pair, quantity)
        price_str = self._fmt_price(pair, price)
        logger.info("Placing LIMIT BUY  %s  qty=%s  price=%s", pair, qty_str, price_str)
        return self._call(
            self._client.place_order,
            pair=pair, side="BUY", type="LIMIT",
            quantity=qty_str, price=float(price_str),
        )

    def place_limit_sell(self, pair: str, quantity: float, price: float) -> dict:
        """Place a limit SELL order with exchange-correct precision."""
        qty_str   = self._fmt_quantity(pair, quantity)
        price_str = self._fmt_price(pair, price)
        logger.info("Placing LIMIT SELL %s  qty=%s  price=%s", pair, qty_str, price_str)
        return self._call(
            self._client.place_order,
            pair=pair, side="SELL", type="LIMIT",
            quantity=qty_str, price=float(price_str),
        )

    def place_market_buy(self, pair: str, quantity: float) -> dict:
        """Place a market BUY order."""
        qty_str = self._fmt_quantity(pair, quantity)
        logger.info("Placing MARKET BUY  %s  qty=%s", pair, qty_str)
        return self._call(
            self._client.place_order,
            pair=pair, side="BUY", type="MARKET", quantity=qty_str,
        )

    def place_market_sell(self, pair: str, quantity: float) -> dict:
        """Place a market SELL order."""
        qty_str = self._fmt_quantity(pair, quantity)
        logger.info("Placing MARKET SELL %s  qty=%s", pair, qty_str)
        return self._call(
            self._client.place_order,
            pair=pair, side="SELL", type="MARKET", quantity=qty_str,
        )

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def get_open_orders(self, pair: Optional[str] = None) -> list[dict]:
        """Return a list of currently pending/open orders."""
        try:
            result = self._call(self._client.query_order, pair=pair, pending_only=True)
            if result.get("Success") is False:
                err = (result.get("ErrMsg") or "").lower()
                if "no order" in err:
                    return []   # Normal — no pending orders exist
                logger.debug("get_open_orders: %s", result.get("ErrMsg"))
                return []
            return result.get("OrderMatched") or []
        except Exception as exc:
            logger.error("get_open_orders failed: %s", exc)
            return []

    def cancel_order(self, order_id: str) -> dict:
        """
        Cancel an open order by ID.

        Assumes Roostoo exposes POST /v3/cancel_order with an order_id param.
        Adjust the endpoint path if the live API differs.
        """
        try:
            return self._call(
                self._client._request,
                "POST", "/v3/cancel_order",
                signed=True,
                params={"order_id": str(order_id)},
            )
        except Exception as exc:
            logger.error("cancel_order(%s) raised: %s", order_id, exc)
            return {"Success": False, "error": str(exc)}

    def get_order_status(self, order_id: str) -> dict:
        """Return the status dict for a specific order, or {} if not found."""
        try:
            result = self._call(self._client.query_order, order_id=order_id)
            orders = result.get("OrderMatched") or []
            return orders[0] if orders else {}
        except Exception as exc:
            logger.error("get_order_status(%s) raised: %s", order_id, exc)
            return {}
