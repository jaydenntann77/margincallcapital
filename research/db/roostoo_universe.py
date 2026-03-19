"""
Fetch the live Roostoo mock-exchange universe from the public API.

Documented in: https://github.com/roostoo/Roostoo-API-Documents
Endpoint: GET /v3/exchangeInfo → TradePairs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

DEFAULT_EXCHANGE_INFO_URL = "https://mock-api.roostoo.com/v3/exchangeInfo"


def fetch_exchange_info(url: str = DEFAULT_EXCHANGE_INFO_URL, timeout: float = 60) -> dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def trade_pairs_to_bases(data: dict[str, Any]) -> list[tuple[str, str]]:
    """
    Return sorted list of (pair, base_coin) for USD-quoted pairs only.
    e.g. ("BTC/USD", "BTC")
    """
    pairs = data.get("TradePairs") or {}
    out: list[tuple[str, str]] = []
    for pair_key, meta in pairs.items():
        if not isinstance(meta, dict):
            continue
        if not pair_key.endswith("/USD"):
            continue
        coin = meta.get("Coin") or pair_key.split("/")[0]
        out.append((pair_key, str(coin)))
    out.sort(key=lambda x: x[0])
    return out


def save_universe_snapshot(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_universe_snapshot(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
