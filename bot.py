#!/usr/bin/env python3
"""
Roostoo Crypto Trading Bot — main entry point.

Usage
-----
  Live trading:
      python bot.py

  Dry-run (signals logged, no orders placed):
      python bot.py --dry-run

Shutdown
--------
  Ctrl-C  or  kill <pid>   →  finishes the current cycle then exits cleanly.
  All open limit orders remain on the exchange; run again to resume tracking.
"""
from __future__ import annotations

import argparse
import signal as _signal
import time
import pandas as pd

# ── Configure logging FIRST so our root-logger setup wins ───────────────────
from logger import get_logger

logger = get_logger("Bot")

from client                import RoostooClient
from strategy              import Signal
from risk_manager          import RiskManager
from signals.long_momentum import LongOnlyMomentumSignal
import config as cfg


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested: bool = False


def _handle_shutdown(sig: int, frame) -> None:  # type: ignore[type-arg]
    global _shutdown_requested
    try:
        name = _signal.Signals(sig).name
    except ValueError:
        name = str(sig)
    logger.info("Shutdown signal received (%s). Finishing current cycle…", name)
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# Order execution  (weight-based — quantity computed by caller)
# ---------------------------------------------------------------------------

def _execute_order(
    api:      RoostooClient,
    risk:     RiskManager,
    pair:     str,
    side:     Signal,
    quantity: float,
    price:    float,
    dry_run:  bool,
) -> None:
    """Validate quantity via risk checks, then place a limit order with market fallback."""
    can_trade, reason, final_qty = risk.check_order(pair, side, quantity, price)

    if not can_trade:
        logger.info("[SKIP]   %-12s %-4s — %s", pair, side.value, reason)
        return

    # Nudge limit price slightly to improve fill probability
    if side == Signal.BUY:
        limit_price = price * (1.0 + cfg.LIMIT_ORDER_OFFSET)
    else:
        limit_price = price * (1.0 - cfg.LIMIT_ORDER_OFFSET)

    logger.info(
        "[SIGNAL] %-12s %-4s | qty=%.6f | last=%.5f | limit=%.5f",
        pair, side.value, final_qty, price, limit_price,
    )

    if dry_run:
        logger.info("[DRY RUN] Order not sent — simulation only.")
        return

    # ── Attempt limit order ─────────────────────────────────────────────
    if side == Signal.BUY:
        result = api.place_limit_buy(pair, final_qty, limit_price)
    else:
        result = api.place_limit_sell(pair, final_qty, limit_price)

    if result.get("Success"):
        detail   = result.get("OrderDetail") or {}
        order_id = str(detail.get("OrderID", "unknown"))
        logger.info(
            "[ORDER]  LIMIT %-4s %-12s | id=%-12s | qty=%.6f | price=%.5f",
            side.value, pair, order_id, final_qty, limit_price,
        )
        risk.track_order(pair, order_id, "LIMIT")
        if side == Signal.BUY:
            risk.record_entry(pair, price)
        else:
            risk.clear_entry(pair)
        return

    # ── Limit order failed → fall back to market order ──────────────────
    err = result.get("ErrMsg") or result.get("error") or "unknown error"
    logger.warning("[WARN]   Limit order failed for %s (%s) — attempting market order.", pair, err)

    if side == Signal.BUY:
        market_result = api.place_market_buy(pair, final_qty)
    else:
        market_result = api.place_market_sell(pair, final_qty)

    if market_result.get("Success"):
        detail   = market_result.get("OrderDetail") or {}
        order_id = str(detail.get("OrderID", "unknown"))
        logger.info(
            "[ORDER]  MARKET %-4s %-12s | id=%-12s | qty=%.6f",
            side.value, pair, order_id, final_qty,
        )
        if side == Signal.BUY:
            risk.record_entry(pair, price)
        else:
            risk.clear_entry(pair)
    else:
        err2 = market_result.get("ErrMsg") or market_result.get("error") or "unknown"
        logger.error("[ERROR]  Market order also failed for %s: %s", pair, err2)


# ---------------------------------------------------------------------------
# Stop-loss execution
# ---------------------------------------------------------------------------

def _execute_stop_loss(
    api:     RoostooClient,
    risk:    RiskManager,
    pair:    str,
    price:   float,
    dry_run: bool,
) -> None:
    """Immediately market-sell the full position when stop-loss is triggered."""
    portfolio = api.get_portfolio_value()
    asset     = pair.split("/")[0]
    quantity  = portfolio.get("asset_values", {}).get(asset, {}).get("amount", 0.0)

    if quantity <= 0:
        logger.warning("[STOP-LOSS] %s triggered but no holdings found — clearing entry.", pair)
        risk.clear_entry(pair)
        return

    logger.warning(
        "[STOP-LOSS] Executing MARKET SELL %s | qty=%.6f | price=%.5f",
        pair, quantity, price,
    )

    if dry_run:
        logger.info("[DRY RUN] Stop-loss market sell not sent — simulation only.")
        risk.clear_entry(pair)
        return

    result = api.place_market_sell(pair, quantity)
    if result.get("Success"):
        detail   = result.get("OrderDetail") or {}
        order_id = str(detail.get("OrderID", "unknown"))
        logger.warning(
            "[STOP-LOSS] FILLED — %s | id=%s | qty=%.6f | price=%.5f",
            pair, order_id, quantity, price,
        )
        risk.clear_entry(pair)
    else:
        err = result.get("ErrMsg") or result.get("error") or "unknown"
        logger.error("[STOP-LOSS] Market sell FAILED for %s: %s — will retry next cycle.", pair, err)


# ---------------------------------------------------------------------------
# Stale-order cleanup
# ---------------------------------------------------------------------------

def _cancel_stale_orders(
    api:     RoostooClient,
    risk:    RiskManager,
    dry_run: bool,
) -> None:
    """Cancel limit orders that have not filled within LIMIT_ORDER_TIMEOUT."""
    for pair, order_id in risk.get_stale_orders():
        logger.info(
            "[STALE]  Cancelling order %s for %s (timed out after %ds)",
            order_id, pair, cfg.LIMIT_ORDER_TIMEOUT,
        )
        if not dry_run:
            result = api.cancel_order(order_id)
            if result.get("Success"):
                logger.info("[CANCEL] Order %s cancelled successfully.", order_id)
            else:
                err = result.get("ErrMsg") or result.get("error") or "unknown"
                logger.warning("[CANCEL] Failed to cancel %s: %s", order_id, err)
        risk.remove_order(pair)


# ---------------------------------------------------------------------------
# Portfolio snapshot
# ---------------------------------------------------------------------------

def _log_portfolio(api: RoostooClient) -> None:
    """Log a concise portfolio summary line."""
    try:
        pf        = api.get_portfolio_value()
        positions = list(pf.get("asset_values", {}).keys())
        logger.info(
            "[PORTFOLIO] Total=$%.2f | Cash=$%.2f | Positions=%s",
            pf["total_usd"], pf["usd_cash"],
            positions if positions else "none",
        )
    except Exception as exc:
        logger.warning("[PORTFOLIO] Could not fetch status: %s", exc)


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _validate_pairs(api: RoostooClient, pairs: list[str]) -> list[str]:
    """Filter configured pairs to those that exist and have CanTrade=True."""
    valid   = [p for p in pairs if api.validate_pair(p)]
    skipped = sorted(set(pairs) - set(valid))
    if skipped:
        logger.warning("Pairs not found / not tradeable on exchange: %s", skipped)
    logger.info("Active trading pairs (%d): %s", len(valid), valid)
    return valid


def _sync_open_orders(api: RoostooClient, risk: RiskManager) -> None:
    """Register any open orders from a previous run with the risk manager."""
    orders = api.get_open_orders()
    if not orders:
        logger.info("[SYNC]   No pre-existing open orders found.")
        return
    for order in orders:
        pair     = str(order.get("Pair") or order.get("pair") or "")
        order_id = str(order.get("OrderID") or order.get("order_id") or "")
        if pair and order_id:
            risk.track_order(pair, order_id, "LIMIT")
            logger.info("[SYNC]   Registered pre-existing order: %s  id=%s", pair, order_id)


def _seed_price_history(api: RoostooClient, pairs: list[str]) -> pd.DataFrame:
    """
    Pre-load price_history from Binance's public API so the momentum strategy
    is fully warmed up from the very first rebalance.

    Falls back gracefully — pairs with no Binance data start with NaN and
    fill in from live prices over time.
    """
    logger.info(
        "Seeding price history from Binance (%d bars × %d pairs)…",
        cfg.MOMENTUM_LOOKBACK_BARS, len(pairs),
    )
    frames: list[pd.Series] = []
    for pair in pairs:
        series = api.get_binance_history(pair, interval="4h", limit=cfg.MOMENTUM_LOOKBACK_BARS)
        if not series.empty:
            frames.append(series)
        else:
            logger.warning("[SEED]   No Binance history for %s — will fill from live prices.", pair)

    if not frames:
        logger.warning("[SEED]   No historical data loaded for any pair — signals delayed until warmup.")
        return pd.DataFrame()

    price_history = pd.concat(frames, axis=1)
    price_history.columns = [s.name for s in frames]
    # Ensure all configured pairs are present (fill missing with NaN)
    for pair in pairs:
        if pair not in price_history.columns:
            price_history[pair] = float("nan")

    logger.info(
        "[SEED]   Loaded %d bars for %d / %d pairs.",
        len(price_history), len(frames), len(pairs),
    )
    return price_history


# ---------------------------------------------------------------------------
# Momentum rebalance
# ---------------------------------------------------------------------------

def _process_momentum_rebalance(
    api:            RoostooClient,
    risk:           RiskManager,
    strategy:       LongOnlyMomentumSignal,
    current_prices: dict[str, float],
    price_history:  pd.DataFrame,
    dry_run:        bool,
) -> pd.DataFrame:
    """
    Append the latest prices as a new bar, generate target weights, and
    execute BUY / SELL orders to rebalance toward those weights.

    Sells are executed before buys so that freed cash is available for buys
    within the same rebalance cycle.

    Returns the updated price_history so the caller can persist it.
    """
    # 1. Append new bar and trim history window
    new_row = pd.DataFrame([current_prices], index=[pd.Timestamp.now()])
    price_history = pd.concat([price_history, new_row]).tail(cfg.MOMENTUM_LOOKBACK_BARS)

    # 2. Warmup guard — need enough bars for vol/trend EMAs to converge
    if len(price_history) < cfg.MIN_WARMUP_BARS:
        logger.info(
            "[MOMENTUM] Warming up — %d / %d bars collected.",
            len(price_history), cfg.MIN_WARMUP_BARS,
        )
        return price_history

    # 3. Generate target weights (latest row of the weight DataFrame)
    target_weights = strategy.generate_weights(price_history).iloc[-1]
    active = {p: f"{w:.4f}" for p, w in target_weights.items() if w > 0}
    logger.info("[MOMENTUM] Target weights: %s", active if active else "all cash")

    # 4. Fetch current portfolio once (used for current-weight calculation)
    portfolio = api.get_portfolio_value()
    total_usd = portfolio.get("total_usd", 0.0)
    if total_usd <= 0:
        logger.warning("[MOMENTUM] Portfolio value is zero — skipping rebalance.")
        return price_history

    # 5. Compute current weights from live holdings
    current_weights: dict[str, float] = {}
    for pair in target_weights.index:
        asset = pair.split("/")[0]
        val   = portfolio.get("asset_values", {}).get(asset, {}).get("value_usd", 0.0)
        current_weights[pair] = val / total_usd

    # 6. Compute weight deltas and bucket into sells / buys
    sells: list[tuple[str, float, float]] = []
    buys:  list[tuple[str, float, float]] = []

    for pair, target_w in target_weights.items():
        price     = current_prices.get(pair)
        if not price:
            continue
        current_w = current_weights.get(pair, 0.0)
        delta     = target_w - current_w

        if delta < -cfg.REBALANCE_THRESHOLD:
            quantity = abs(delta) * total_usd / price
            sells.append((pair, quantity, price))
            logger.debug("[REBALANCE] SELL %s  delta=%.4f  qty=%.6f", pair, delta, quantity)
        elif delta > cfg.REBALANCE_THRESHOLD:
            quantity = delta * total_usd / price
            buys.append((pair, quantity, price))
            logger.debug("[REBALANCE] BUY  %s  delta=%.4f  qty=%.6f", pair, delta, quantity)

    # 7. Execute sells first (frees cash), then buys
    for pair, quantity, price in sells:
        _execute_order(api, risk, pair, Signal.SELL, quantity, price, dry_run)
    for pair, quantity, price in buys:
        _execute_order(api, risk, pair, Signal.BUY, quantity, price, dry_run)

    if not sells and not buys:
        logger.info("[MOMENTUM] Portfolio already within rebalance threshold — no trades needed.")

    return price_history


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Parse args ───────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Roostoo Momentum Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Log signals without placing orders")
    args    = parser.parse_args()
    dry_run = args.dry_run

    _signal.signal(_signal.SIGINT,  _handle_shutdown)
    _signal.signal(_signal.SIGTERM, _handle_shutdown)

    if dry_run:
        logger.info("DRY-RUN mode enabled — no orders will be placed.")

    # ── Initialise components ────────────────────────────────────────────
    api  = RoostooClient()
    risk = RiskManager(api)
    strategy = LongOnlyMomentumSignal(
        fast_span            = cfg.FAST_SPAN,
        slow_span            = cfg.SLOW_SPAN,
        vol_span             = cfg.VOL_SPAN,
        z_score_threshold    = cfg.Z_SCORE_THRESHOLD,
        top_n                = cfg.TOP_N,
        trend_filter_span    = cfg.TREND_FILTER_SPAN,
        target_vol           = cfg.TARGET_VOL,
        annualization_factor = cfg.ANNUALIZATION_FACTOR,
    )

    active_pairs = _validate_pairs(api, cfg.TRADING_PAIRS)
    _sync_open_orders(api, risk)

    # ── Seed price history from Binance public API ───────────────────────
    price_history: pd.DataFrame = _seed_price_history(api, active_pairs)

    last_momentum_ts = 0.0

    _log_portfolio(api)

    # ── Main loop ────────────────────────────────────────────────────────
    cycle: int = 0
    while not _shutdown_requested:
        cycle_start  = time.monotonic()
        current_time = time.time()
        cycle += 1

        # Fetch all prices once per cycle
        current_prices: dict[str, float] = {
            p: price
            for p in active_pairs
            if (price := api.get_price(p)) is not None
        }

        # 1. Stop-loss check — runs every 10s on live prices
        for pair, price in current_prices.items():
            if risk.check_stop_loss(pair, price):
                _execute_stop_loss(api, risk, pair, price, dry_run)

        # 2. Momentum rebalance — runs every 4 hours
        if current_time - last_momentum_ts >= cfg.MOMENTUM_INTERVAL_SECONDS:
            logger.info("─── 4-Hour Momentum Rebalance (cycle %d) ───", cycle)
            price_history    = _process_momentum_rebalance(
                api, risk, strategy, current_prices, price_history, dry_run,
            )
            last_momentum_ts = current_time

        # 3. Housekeeping
        _cancel_stale_orders(api, risk, dry_run)
        if cycle % 10 == 0:
            _log_portfolio(api)

        # 4. Sleep for the remainder of the polling interval
        elapsed    = time.monotonic() - cycle_start
        sleep_time = max(0.0, cfg.POLL_INTERVAL_SECONDS - elapsed)
        logger.info("Cycle %d done in %.1fs — next poll in %.1fs.", cycle, elapsed, sleep_time)
        if not _shutdown_requested and sleep_time > 0:
            time.sleep(sleep_time)

    logger.info("Bot shut down cleanly after %d cycle(s).", cycle)


if __name__ == "__main__":
    main()
