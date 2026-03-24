"""
Signal enum shared across the trading system.

BUY  — open / increase a position
SELL — close / reduce a position
HOLD — no action this cycle
"""
from __future__ import annotations

from enum import Enum


class Signal(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
