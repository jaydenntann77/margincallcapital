"""
Load multi-interval OHLCV for the *live* Roostoo API universe into DuckDB.

Live in: research/db/ — run from repo root.

1. Roostoo GET /v3/exchangeInfo → TradePairs
2. Binance Vision 1m (monthly + daily fallback for current month)
3. Coinbase 1m fallback
4. Resample 5m, 15m, 1h; optional 5s from 1s

Usage:
    python research/db/load_1m_roostoo_universe.py --db data/research_roostoo_1m.duckdb --months 3
    python research/db/load_1m_roostoo_universe.py --no-fetch-5s
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests

_DB_DIR = Path(__file__).resolve().parent
if str(_DB_DIR) not in sys.path:
    sys.path.insert(0, str(_DB_DIR))

from binance_vision import (  # noqa: E402
    download_binance_klines_calendar_month,
    month_range_calendar,
)
from roostoo_universe import (  # noqa: E402
    DEFAULT_EXCHANGE_INFO_URL,
    fetch_exchange_info,
    save_universe_snapshot,
    trade_pairs_to_bases,
)

ROOT = Path(__file__).resolve().parent.parent.parent

COINBASE_CANDLES = "https://api.coinbase.com/api/v3/brokerage/market/products"
COINBASE_PRODUCTS = "https://api.coinbase.com/api/v3/brokerage/market/products"
INTERVAL_1M = "1m"
USER_AGENT = {"User-Agent": "margincallcapital-research/1.0"}

BINANCE_BASE_OVERRIDES: dict[str, str] = {}


def fetch_coinbase_usd_product_ids(session: requests.Session) -> set[str]:
    r = session.get(COINBASE_PRODUCTS, timeout=120, headers=USER_AGENT)
    r.raise_for_status()
    j = r.json()
    prods = j.get("products", j) if isinstance(j, dict) else j
    ids: set[str] = set()
    for p in prods:
        if not isinstance(p, dict):
            continue
        pid = p.get("product_id") or ""
        if not pid.endswith("-USD"):
            continue
        if p.get("status") != "online" or p.get("trading_disabled"):
            continue
        ids.add(pid)
    return ids


def download_coinbase_1m(
    session: requests.Session,
    product_id: str,
    start: datetime,
    end: datetime,
    delay_s: float,
) -> pd.DataFrame | None:
    start_ts = int(start.replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(end.replace(tzinfo=timezone.utc).timestamp())
    window = 350 * 60
    cursor_end = end_ts
    rows: list[dict] = []

    while cursor_end > start_ts:
        chunk_start = max(start_ts, cursor_end - window)
        url = f"{COINBASE_CANDLES}/{product_id}/candles"
        params = {
            "start": str(chunk_start),
            "end": str(cursor_end),
            "granularity": "ONE_MINUTE",
        }
        try:
            r = session.get(url, params=params, timeout=60, headers=USER_AGENT)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            print(f"    Coinbase: {e}")
            return None

        candles = data.get("candles") or []
        for c in candles:
            rows.append(
                {
                    "ts": datetime.fromtimestamp(int(c["start"]), tz=timezone.utc),
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": float(c["volume"]),
                }
            )

        cursor_end = chunk_start
        time.sleep(delay_s)

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    return df


def filter_since_window(combined: pd.DataFrame, start_window: datetime) -> pd.DataFrame:
    ts_utc = pd.to_datetime(combined["ts"], utc=True)
    sw = pd.Timestamp(start_window)
    if sw.tzinfo is None:
        sw = sw.tz_localize("UTC")
    else:
        sw = sw.tz_convert("UTC")
    out = combined.loc[ts_utc >= sw].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True).dt.tz_localize(None)
    return out


def fetch_binance_1s_months(
    binance_kline: str, months: list[tuple[int, int]], start_window: datetime
) -> pd.DataFrame | None:
    all_dfs: list[pd.DataFrame] = []
    for y, m in months:
        df = download_binance_klines_calendar_month(binance_kline, y, m, "1s")
        if df is not None and not df.empty:
            all_dfs.append(df)
    if not all_dfs:
        return None
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ts"]).sort_values("ts")
    return filter_since_window(combined, start_window)


def ensure_ohlcv_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv (
            ts TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            symbol VARCHAR,
            interval VARCHAR
        )
        """
    )


def _ohlcv_indexed(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["ts", "open", "high", "low", "close", "volume"]].copy()
    out = out.set_index("ts")
    if hasattr(out.index, "tz") and out.index.tz is not None:
        out = out.tz_localize(None, ambiguous="infer")
    return out.sort_index()


def resample_bars(idx: pd.DataFrame, rule: str) -> pd.DataFrame:
    bar = (
        idx.resample(rule, label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(how="any")
        .reset_index()
    )
    return bar


def insert_all_intervals(
    con: duckdb.DuckDBPyConnection,
    df_1m: pd.DataFrame,
    ohlcv_sym: str,
    df_1s: pd.DataFrame | None,
) -> tuple[int, int]:
    for itv in ("5s", "1m", "5m", "15m", "1h"):
        con.execute("DELETE FROM ohlcv WHERE symbol = ? AND interval = ?", [ohlcv_sym, itv])

    def _insert(df: pd.DataFrame, interval: str) -> None:
        if df is None or df.empty:
            return
        d = df.copy()
        d["symbol"] = ohlcv_sym
        d["interval"] = interval
        con.register("_t", d)
        con.execute(
            "INSERT INTO ohlcv SELECT ts, open, high, low, close, volume, symbol, interval FROM _t"
        )
        con.unregister("_t")

    _insert(df_1m, "1m")
    n1m = len(df_1m)

    idx1m = _ohlcv_indexed(df_1m)
    _insert(resample_bars(idx1m, "5min"), "5m")
    _insert(resample_bars(idx1m, "15min"), "15m")
    _insert(resample_bars(idx1m, "1h"), "1h")

    n5s = 0
    if df_1s is not None and not df_1s.empty:
        idx1s = _ohlcv_indexed(df_1s)
        df5 = resample_bars(idx1s, "5s")
        n5s = len(df5)
        _insert(df5, "5s")

    return n1m, n5s


def main() -> None:
    parser = argparse.ArgumentParser(description="Load 1m OHLCV for Roostoo API universe")
    parser.add_argument("--db", default="data/research_roostoo_1m.duckdb", help="Output DuckDB path")
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--roostoo-url", default=DEFAULT_EXCHANGE_INFO_URL)
    parser.add_argument("--universe-out", default="data/roostoo_universe.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--coinbase-delay", type=float, default=0.08)
    parser.add_argument("--prefer", choices=("auto", "binance", "coinbase"), default="auto")
    parser.add_argument("--no-fetch-5s", action="store_true")
    args = parser.parse_args()

    db_path = ROOT / args.db if not Path(args.db).is_absolute() else Path(args.db)
    uni_path = ROOT / args.universe_out if not Path(args.universe_out).is_absolute() else Path(args.universe_out)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching Roostoo exchangeInfo...")
    info = fetch_exchange_info(args.roostoo_url)
    save_universe_snapshot(info, uni_path)
    print(f"  Saved universe snapshot → {uni_path}")

    pairs = trade_pairs_to_bases(info)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"  USD pairs to load: {len(pairs)}")

    now = datetime.now(timezone.utc)
    start_window = now - timedelta(days=30 * args.months)

    session = requests.Session()
    coinbase_ids: set[str] = set()
    if args.prefer in ("auto", "coinbase"):
        print("Fetching Coinbase USD product list (for fallback)...")
        coinbase_ids = fetch_coinbase_usd_product_ids(session)
        print(f"  {len(coinbase_ids)} online *-USD products")

    con = duckdb.connect(str(db_path))
    ensure_ohlcv_table(con)

    con.execute("DROP TABLE IF EXISTS roostoo_load_meta")
    con.execute(
        """
        CREATE TABLE roostoo_load_meta (
            base VARCHAR,
            pair VARCHAR,
            source VARCHAR,
            rows_1m INTEGER,
            rows_5s INTEGER,
            updated_at TIMESTAMP
        )
        """
    )

    months = month_range_calendar(args.months)
    print(f"  Calendar months (UTC, oldest→newest): {months}")
    stats: list[tuple[str, str, str, int, int]] = []

    for pair_key, base in pairs:
        ohlcv_sym = f"{base}-USD"
        binance_kline = BINANCE_BASE_OVERRIDES.get(base, f"{base}USDT")
        print(f"\n{pair_key} → Vision `{binance_kline}` / Coinbase `{base}-USD`")

        combined: pd.DataFrame | None = None
        source = ""

        if args.prefer in ("auto", "binance"):
            all_dfs: list[pd.DataFrame] = []
            for y, m in months:
                df = download_binance_klines_calendar_month(
                    binance_kline, y, m, INTERVAL_1M, verbose_label="Vision 1m"
                )
                if df is not None and not df.empty:
                    all_dfs.append(df)
            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                combined = combined.drop_duplicates(subset=["ts"]).sort_values("ts")
                combined = filter_since_window(combined, start_window)
                source = "binance_vision"

        if combined is None or combined.empty:
            if args.prefer == "binance":
                print("  Skip: no Binance Vision data (--prefer binance)")
                stats.append((base, pair_key, "skipped_no_binance", 0, 0))
                continue
            pid = f"{base}-USD"
            if pid not in coinbase_ids:
                print(f"  Skip: no Binance data and `{pid}` not on Coinbase")
                stats.append((base, pair_key, "skipped_no_source", 0, 0))
                continue
            print(f"  Fallback: Coinbase {pid} (may take a while)...")
            combined = download_coinbase_1m(session, pid, start_window, now, delay_s=args.coinbase_delay)
            source = "coinbase" if combined is not None and not combined.empty else ""

        if combined is None or combined.empty:
            print("  No rows loaded")
            stats.append((base, pair_key, "skipped_empty", 0, 0))
            continue

        df_1s: pd.DataFrame | None = None
        if (
            not args.no_fetch_5s
            and source == "binance_vision"
            and args.prefer in ("auto", "binance")
        ):
            print("  Fetching 1s (Vision) for 5s bars…")
            df_1s = fetch_binance_1s_months(binance_kline, months, start_window)
            if df_1s is None or df_1s.empty:
                print("  (no 1s data — 5s skipped; 1m/5m/15m/1h still stored)")
                df_1s = None
        elif source == "coinbase" and not args.no_fetch_5s:
            print("  Note: Coinbase path is 1m-only — cannot build true 5s (skipped)")

        n1m, n5s = insert_all_intervals(con, combined, ohlcv_sym, df_1s)
        extra = ", 5m, 15m, 1h resampled"
        if n5s:
            extra += f"; 5s: {n5s:,} rows (from 1s)"
        print(f"  ✓ {source}: {n1m:,} rows 1m{extra}")
        stats.append((base, pair_key, source, n1m, n5s))
        con.execute(
            "INSERT INTO roostoo_load_meta VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [base, pair_key, source, n1m, n5s],
        )

    con.close()

    summary = {
        "db": str(db_path),
        "universe_file": str(uni_path),
        "pairs_requested": len(pairs),
        "loaded": sum(1 for s in stats if s[3] > 0),
        "by_source": {},
    }
    for _, _, src, n1m, _n5 in stats:
        if n1m <= 0:
            continue
        summary["by_source"][src] = summary["by_source"].get(src, 0) + 1

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nDone. Database: {db_path.resolve()}")


if __name__ == "__main__":
    main()
