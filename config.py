"""
Central configuration for the Roostoo trading bot.

All secrets are loaded from environment variables (via .env).
Everything else is a tunable constant — edit here, nowhere else.

The original API-credential exports (API_KEY, API_SECRET, BASE_URL) are
preserved so that the existing client.py continues to work without changes.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API Credentials  (loaded from .env — NEVER hardcode these)
# ---------------------------------------------------------------------------
API_KEY    = os.getenv("ROOSTOO_API_KEY", "")
API_SECRET = os.getenv("ROOSTOO_API_SECRET", "")
BASE_URL   = os.getenv("ROOSTOO_BASE_URL", "https://mock.roostoo.com")

# ---------------------------------------------------------------------------
# Trading Universe
# Pairs are validated against the live exchange at startup.
# Invalid / non-tradeable entries are skipped automatically.
# All pairs use the /USD quote currency as per Roostoo's exchange_info.
# ---------------------------------------------------------------------------
TRADING_PAIRS: list[str] = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "BNB/USD",
    "XRP/USD",
    "ADA/USD",
    "DOGE/USD",
    "LINK/USD",
    "AVAX/USD",
    "SUI/USD",
]

# ---------------------------------------------------------------------------
# EMA Strategy Parameters
# ---------------------------------------------------------------------------
FAST_EMA_PERIOD: int = 9     # Short-term EMA period (bars) — 9 × 10s = 1.5min
SLOW_EMA_PERIOD: int = 30   # Long-term  EMA period (bars) — 30 × 10s = 5min

# Number of bars required before the strategy emits a real signal.
# Allows both EMAs to converge from their seed value (first observed price).
MIN_HISTORY_BARS: int = SLOW_EMA_PERIOD + 1   # 31 bars = ~5.2 min warmup

# ---------------------------------------------------------------------------
# Position Sizing & Risk
# ---------------------------------------------------------------------------
TRADE_FRACTION: float      = 0.0005 # Fraction of total portfolio per BUY trade (5 bps / 0.05%)
MAX_POSITION_FRAC: float   = 0.10   # Max fraction of portfolio in any single asset
SELL_FRACTION: float       = 1.0    # Fraction of current holdings sold per SELL signal
MIN_ORDER_VALUE_USD: float = 15.0   # Minimum USD value per order (buffer above MiniOrder=1)
STOP_LOSS_PCT: float       = 0.03   # 3% drop from entry price triggers an immediate market sell

# ---------------------------------------------------------------------------
# Commission Rates
# ---------------------------------------------------------------------------
TAKER_FEE: float = 0.001   # 0.1%  — market orders
MAKER_FEE: float = 0.0005  # 0.05% — limit orders resting in the book

# ---------------------------------------------------------------------------
# Limit Order Settings
# ---------------------------------------------------------------------------
# Price offset applied to limit orders to improve fill probability.
#   BUY  limit = last_price * (1 + LIMIT_ORDER_OFFSET)  → slightly above market
#   SELL limit = last_price * (1 - LIMIT_ORDER_OFFSET)  → slightly below market
# Set to 0.0 to post exactly at last price (may rest as maker).
LIMIT_ORDER_OFFSET: float = 0.001    # 0.1% offset

# Seconds after which an unfilled limit order is cancelled and a market-order
# fallback is attempted on the next signal.
LIMIT_ORDER_TIMEOUT: int = 120

# ---------------------------------------------------------------------------
# Signal Cooldown
# Prevents re-entering the same signal direction for the same pair too quickly.
# ---------------------------------------------------------------------------
SIGNAL_COOLDOWN_SECONDS: int = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------
POLL_INTERVAL_SECONDS: int = 10    # Seconds between price-poll cycles

# ---------------------------------------------------------------------------
# Retry / Exponential Backoff (for transient network failures)
# ---------------------------------------------------------------------------
MAX_RETRIES: int     = 3
RETRY_BACKOFF: float = 2.0   # Base seconds (doubles each attempt)
RETRY_JITTER: float  = 0.5   # Uniform random fraction added to avoid thundering herd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE:  str = "bot.log"
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# --- Strategy Partitioning ---
MOMENTUM_CAPITAL_SHARE: float = 0.5  # 50% of total equity
EMA_CAPITAL_SHARE: float      = 0.5  # 50% of total equity

# --- 4-Hour Momentum Settings ---
MOMENTUM_INTERVAL_SECONDS: int = 14400  # 4 hours (60 * 60 * 4)
MOMENTUM_LOOKBACK_BARS: int    = 300    # Enough for 250 vol_span