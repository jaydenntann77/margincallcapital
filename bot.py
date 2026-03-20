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
    else:
        err2 = market_result.get("ErrMsg") or market_result.get("error") or "unknown"
        logger.error("[ERROR]  Market order also failed for %s: %s", pair, err2)


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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roostoo crypto trading bot — EMA crossover strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log signals and risk decisions but do NOT place real orders.",
    )
    args    = parser.parse_args()
    dry_run = args.dry_run

    # Register OS-level shutdown handlers
    _signal.signal(_signal.SIGINT,  _handle_shutdown)
    _signal.signal(_signal.SIGTERM, _handle_shutdown)

    logger.info("=" * 64)
    logger.info("  Roostoo Trading Bot — Starting Up")
    logger.info("  Mode            : %s", "DRY RUN (no orders)" if dry_run else "LIVE TRADING")
    logger.info("  EMA periods     : fast=%d  slow=%d", cfg.FAST_EMA_PERIOD, cfg.SLOW_EMA_PERIOD)
    logger.info("  Poll interval   : %ds", cfg.POLL_INTERVAL_SECONDS)
    logger.info("  Trade fraction  : %.0f%%  |  Max position: %.0f%%",
                cfg.TRADE_FRACTION * 100, cfg.MAX_POSITION_FRAC * 100)
    logger.info("  Configured pairs: %s", cfg.TRADING_PAIRS)
    logger.info("=" * 64)

    # Validate credentials
    if not cfg.API_KEY or not cfg.API_SECRET:
        logger.critical(
            "ROOSTOO_API_KEY and/or ROOSTOO_API_SECRET are not set. "
            "Copy .env.example to .env and add your credentials, then retry."
        )
        sys.exit(1)

    # Initialise components
    try:
        api      = RoostooClient()
        strategy = EMACrossoverStrategy(cfg.FAST_EMA_PERIOD, cfg.SLOW_EMA_PERIOD)
        risk     = RiskManager(api)
    except Exception as exc:
        logger.critical("Failed to initialise components: %s", exc, exc_info=True)
        sys.exit(1)

    # Verify connectivity
    logger.info("Verifying API connectivity…")
    try:
        info = api.get_exchange_info()
        if info.get("IsRunning"):
            logger.info("Exchange is running — connectivity OK.")
        else:
            logger.warning("Exchange reported IsRunning=False — proceeding anyway.")
    except Exception as exc:
        logger.critical("Cannot reach Roostoo API: %s", exc)
        sys.exit(1)

    # Validate and filter trading pairs
    active_pairs = _validate_pairs(api, cfg.TRADING_PAIRS)
    if not active_pairs:
        logger.critical("No valid trading pairs found. Check TRADING_PAIRS in config.py.")
        sys.exit(1)

    # Sync any open orders from a previous run
    _sync_open_orders(api, risk)

    # Log starting portfolio
    _log_portfolio(api)

    cycle: int = 0
    while not _shutdown_requested:
        cycle_start = time.monotonic()
        cycle      += 1
        logger.info("─── Cycle %d ───", cycle)

        # 1. Cancel stale limit orders
        try:
            _cancel_stale_orders(api, risk, dry_run)
        except Exception as exc:
            logger.error("Stale-order cleanup error: %s", exc, exc_info=True)

        # 2. Process each pair
        for pair in active_pairs:
            if _shutdown_requested:
                break
            try:
                price = api.get_price(pair)
                if price is None:
                    logger.warning("[%s] Price unavailable — skipping this cycle.", pair)
                    continue

                sig = strategy.update(pair, price)
                ema = strategy.get_ema_values(pair)

                if ema.get("warmed_up"):
                    logger.debug(
                        "[%s] price=%.5f  fast=%.5f  slow=%.5f  spread=%+.5f  → %s",
                        pair, price,
                        ema.get("fast_ema") or 0,
                        ema.get("slow_ema") or 0,
                        ema.get("spread")   or 0,
                        sig.value,
                    )
                else:
                    logger.info(
                        "[%s] Warming up — %d/%d bars collected, holding.",
                        pair,
                        ema.get("bar_count", 0),
                        cfg.SLOW_EMA_PERIOD,
                    )

                if sig != Signal.HOLD:
                    _execute_signal(api, risk, pair, sig, price, dry_run)

            except Exception as exc:
                logger.error("[%s] Unhandled error: %s", pair, exc, exc_info=True)

        # 3. Periodic portfolio snapshot every 10 cycles
        if cycle % 10 == 0:
            _log_portfolio(api)

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
