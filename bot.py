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
import sys
import time
import pandas as pd

# ── Configure logging FIRST so our root-logger setup wins over the
#    logging.basicConfig() call inside client.py ──────────────────────────
from logger import get_logger

logger = get_logger("Bot")

from client       import RoostooClient
from strategy     import EMACrossoverStrategy, Signal
from risk_manager import RiskManager
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
# Order execution
# ---------------------------------------------------------------------------

def _execute_signal(
    api:     RoostooClient,
    risk:    RiskManager,
    pair:    str,
    sig:     Signal,
    price:   float,
    dry_run: bool,
) -> None:
    """Run risk checks, then try a limit order with a market-order fallback."""
    can_trade, reason, quantity = risk.check(pair, sig, price)

    if not can_trade:
        logger.info("[SKIP]   %-12s %-4s — %s", pair, sig.value, reason)
        return

    # Limit price: nudge slightly away from market to improve fill probability
    if sig == Signal.BUY:
        limit_price = price * (1.0 + cfg.LIMIT_ORDER_OFFSET)
    else:
        limit_price = price * (1.0 - cfg.LIMIT_ORDER_OFFSET)

    logger.info(
        "[SIGNAL] %-12s %-4s | qty=%.6f | last_price=%.5f | limit_price=%.5f",
        pair, sig.value, quantity, price, limit_price,
    )

    if dry_run:
        logger.info("[DRY RUN] Order not sent — simulation only.")
        # Record the signal so cooldown logic works in dry-run mode too
        risk.record_signal(pair, sig)
        return

    # ── Attempt limit order ─────────────────────────────────────────────
    if sig == Signal.BUY:
        result = api.place_limit_buy(pair, quantity, limit_price)
    else:
        result = api.place_limit_sell(pair, quantity, limit_price)

    if result.get("Success"):
        detail   = result.get("OrderDetail") or {}
        order_id = str(detail.get("OrderID", "unknown"))
        logger.info(
            "[ORDER]  LIMIT %-4s %-12s | id=%-12s | qty=%.6f | price=%.5f",
            sig.value, pair, order_id, quantity, limit_price,
        )
        risk.record_signal(pair, sig)
        risk.track_order(pair, order_id, "LIMIT")
        if sig == Signal.BUY:
            risk.record_entry(pair, price)
        else:
            risk.clear_entry(pair)
        return

    # ── Limit order failed → fall back to market order ──────────────────
    err = result.get("ErrMsg") or result.get("error") or "unknown error"
    logger.warning(
        "[WARN]   Limit order failed for %s (%s) — attempting market order.", pair, err
    )

    if sig == Signal.BUY:
        market_result = api.place_market_buy(pair, quantity)
    else:
        market_result = api.place_market_sell(pair, quantity)

    if market_result.get("Success"):
        detail   = market_result.get("OrderDetail") or {}
        order_id = str(detail.get("OrderID", "unknown"))
        logger.info(
            "[ORDER]  MARKET %-4s %-12s | id=%-12s | qty=%.6f",
            sig.value, pair, order_id, quantity,
        )
        risk.record_signal(pair, sig)
        if sig == Signal.BUY:
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
        risk.record_signal(pair, Signal.SELL)   # start cooldown so we don't re-buy immediately
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
    """
    Filter the configured pair list to those that actually exist on the
    exchange and have CanTrade=True.
    """
    valid   = [p for p in pairs if api.validate_pair(p)]
    skipped = sorted(set(pairs) - set(valid))
    if skipped:
        logger.warning("Pairs not found / not tradeable on exchange: %s", skipped)
    logger.info("Active trading pairs (%d): %s", len(valid), valid)
    return valid


def _sync_open_orders(api: RoostooClient, risk: RiskManager) -> None:
    """
    On startup, fetch any open orders left from a previous run and register
    them with the risk manager so they are subject to stale-order cleanup.
    """
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

def _process_ema_signals(api, risk, strategy, current_prices, dry_run):
    """Processes the 10s EMA crossover strategy using 50% capital."""
    for pair, price in current_prices.items():
        # Stop-loss check (Global priority)
        if risk.check_stop_loss(pair, price):
            _execute_stop_loss(api, risk, pair, price, dry_run)
            continue

        sig = strategy.update(pair, price)
        if sig != Signal.HOLD:
            # Use cfg.EMA_CAPITAL_SHARE (0.5)
            can_trade, reason, qty = risk.check(pair, sig, price, share=cfg.EMA_CAPITAL_SHARE)
            if can_trade:
                _execute_signal(api, risk, pair, sig, price, dry_run)

def _process_momentum_rebalance(api, risk, strategy, current_prices, price_history, dry_run):
    """Processes the 4-hour cross-sectional momentum using 50% capital."""
    # Update the historical buffer with the latest 4H close
    new_row = pd.DataFrame([current_prices], index=[pd.Timestamp.now()])
    price_history = pd.concat([price_history, new_row]).tail(300)

    # Generate cross-sectional weights
    target_weights = strategy.generate_weights(price_history).iloc[-1]
    
    for pair, weight in target_weights.items():
        price = current_prices.get(pair)
        if not price: continue

        # Logic: If weight > 0, ensure we have a position. If weight == 0, exit.
        if weight > 0:
            # Check risk with cfg.MOMENTUM_CAPITAL_SHARE (0.5)
            can_trade, reason, qty = risk.check(pair, Signal.BUY, price, share=cfg.MOMENTUM_CAPITAL_SHARE)
            if can_trade:
                _execute_signal(api, risk, pair, Signal.BUY, price, dry_run)
        else:
            # Exit position if momentum falls out of Top N or Z-score drops
            can_trade, reason, qty = risk.check(pair, Signal.SELL, price, share=cfg.MOMENTUM_CAPITAL_SHARE)
            if can_trade:
                _execute_signal(api, risk, pair, Signal.SELL, price, dry_run)                
# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    # ... (Argparse and logging setup unchanged) ...
    from signals.long_momentum import LongOnlyMomentumSignal
    
    # Initialise components
    api      = RoostooClient()
    ema_strat = EMACrossoverStrategy(cfg.FAST_EMA_PERIOD, cfg.SLOW_EMA_PERIOD)
    mom_strat = LongOnlyMomentumSignal(vol_span=250, target_vol=0.40) # 4H settings
    risk     = RiskManager(api)

    active_pairs = _validate_pairs(api, cfg.TRADING_PAIRS)
    _sync_open_orders(api, risk)

    # Pre-load 4H historical data for Momentum Strategy
    logger.info("Pre-loading 4-hour historical data...")
    price_history = pd.DataFrame()
    for pair in active_pairs:
        # Assuming api.get_historical_candles is implemented in client.py
        price_history[pair] = api.get_historical_candles(pair, interval="4h", limit=300)

    last_momentum_ts = 0.0
    MOMENTUM_INTERVAL = 14400  # 4 hours in seconds

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Log signals without trading")
    args = parser.parse_args()
    dry_run = args.dry_run

    cycle: int = 0
    while not _shutdown_requested:
        cycle_start = time.monotonic()
        current_time = time.time()
        cycle += 1
        
        # Fetch current market prices once per cycle for both strategies
        current_prices = {p: api.get_price(p) for p in active_pairs if api.get_price(p)}

        # 1. EMA STRATEGY (Every 10s Poll)
        _process_ema_signals(api, risk, ema_strat, current_prices, dry_run)

        # 2. MOMENTUM STRATEGY (Every 4 Hours)
        if current_time - last_momentum_ts >= MOMENTUM_INTERVAL:
            logger.info("─── Executing 4-Hour Momentum Rebalance ───")
            _process_momentum_rebalance(api, risk, mom_strat, current_prices, price_history, dry_run)
            last_momentum_ts = current_time

        # 3. Standard Housekeeping (Stale orders, logging, sleep)
        _cancel_stale_orders(api, risk, dry_run)
        if cycle % 10 == 0: _log_portfolio(api)
        
        # Sleep logic (unchanged)

        # 4. Sleep for the remainder of the polling interval
        elapsed    = time.monotonic() - cycle_start
        sleep_time = max(0.0, cfg.POLL_INTERVAL_SECONDS - elapsed)
        logger.info(
            "Cycle %d done in %.1fs — next poll in %.1fs.",
            cycle, elapsed, sleep_time,
        )
        if not _shutdown_requested and sleep_time > 0:
            time.sleep(sleep_time)

    logger.info("Bot shut down cleanly after %d cycle(s).", cycle)


if __name__ == "__main__":
    main()
