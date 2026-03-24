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
# Position Sizing & Risk
# ---------------------------------------------------------------------------
MAX_POSITION_FRAC: float   = 0.10   # Max fraction of portfolio in any single asset
MIN_ORDER_VALUE_USD: float = 15.0   # Minimum USD value per order (buffer above MiniOrder=1)
STOP_LOSS_PCT: float       = 0.03   # 3% drop from entry price triggers an immediate market sell
REBALANCE_THRESHOLD: float = 0.02   # Min weight drift (2%) required to trigger a rebalance trade

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

# ---------------------------------------------------------------------------
# Momentum Strategy — Signal Parameters
# These mirror LongOnlyMomentumSignal.__init__ defaults; edit here, not there.
# ---------------------------------------------------------------------------
FAST_SPAN:            int   = 5            # Fast EMA span (bars)
SLOW_SPAN:            int   = 20           # Slow EMA span (bars)
VOL_SPAN:             int   = 250          # EWM vol / z-score span (bars)
Z_SCORE_THRESHOLD:    float = 0.9          # Min z-score to qualify for selection
TOP_N:                int   = 3            # Max simultaneous holdings
TREND_FILTER_SPAN:    int   = 200          # Absolute-trend EMA span (bars)
TARGET_VOL:           float = 0.40         # Target annualised portfolio volatility
ANNUALIZATION_FACTOR: float = 365.0 * 6   # 4h bars → 6 bars/day × 365 days

# ---------------------------------------------------------------------------
# Momentum Strategy — Execution Settings
# ---------------------------------------------------------------------------
MOMENTUM_INTERVAL_SECONDS: int = 14400  # Rebalance every 4 hours
MOMENTUM_LOOKBACK_BARS:    int = 300    # Rolling price-history window kept in memory
MIN_WARMUP_BARS:           int = 250    # Bars required before signals are trusted (= VOL_SPAN)