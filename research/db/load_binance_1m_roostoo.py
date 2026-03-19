"""
Legacy: fixed list of Roostoo-style tokens → Binance Vision 1m (+ 1h resample).

Run from repo root:
    python research/db/load_binance_1m_roostoo.py --db data/research_1m.duckdb
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

_DB_DIR = Path(__file__).resolve().parent
if str(_DB_DIR) not in sys.path:
    sys.path.insert(0, str(_DB_DIR))

from binance_vision import download_binance_klines_calendar_month, month_range_calendar  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent

ROOSTOO_TOKENS = [
    "BTC", "ETH", "XRP", "BCH", "LTC", "BNB", "EOS", "TRX", "ATOM", "DOGE",
    "LINK", "ADA", "ZRX", "BAT", "ETC", "ZEC", "DASH", "MATIC",
]
INTERVAL = "1m"
MONTHS_BACK = 3


def binance_symbol(token: str) -> str:
    return f"{token}USDT"


def to_ohlcv_symbol(token: str) -> str:
    return f"{token}-USD"


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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Load 1m Binance Vision data for fixed Roostoo token list")
    parser.add_argument("--db", default="data/research_1m.duckdb")
    parser.add_argument("--months", type=int, default=MONTHS_BACK)
    args = parser.parse_args()

    db_path = ROOT / args.db if not Path(args.db).is_absolute() else Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    ensure_ohlcv_table(con)

    months = month_range_calendar(args.months)
    total_rows = 0

    for token in ROOSTOO_TOKENS:
        bin_sym = binance_symbol(token)
        ohlcv_sym = to_ohlcv_symbol(token)
        print(f"\n{token} ({bin_sym})...")

        all_dfs = []
        for year, month in months:
            df = download_binance_klines_calendar_month(bin_sym, year, month, INTERVAL)
            if df is not None and not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            print(f"  No data for {token}")
            continue

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts"]).sort_values("ts")
        combined["symbol"] = ohlcv_sym
        combined["interval"] = INTERVAL

        con.execute("DELETE FROM ohlcv WHERE symbol = ? AND interval = ?", [ohlcv_sym, INTERVAL])
        con.register("_df", combined)
        con.execute(
            "INSERT INTO ohlcv SELECT ts, open, high, low, close, volume, symbol, interval FROM _df"
        )
        con.unregister("_df")
        total_rows += len(combined)
        print(f"  Inserted {len(combined):,} rows (1m)")

        df_1m = combined.set_index("ts")
        if hasattr(df_1m.index, "tz") and df_1m.index.tz is not None:
            df_1m = df_1m.tz_localize(None, ambiguous="infer")
        resampled = (
            df_1m.resample("1h")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
            .reset_index()
        )
        resampled["symbol"] = ohlcv_sym
        resampled["interval"] = "1h"
        con.execute("DELETE FROM ohlcv WHERE symbol = ? AND interval = '1h'", [ohlcv_sym])
        con.register("_df1h", resampled)
        con.execute(
            "INSERT INTO ohlcv SELECT ts, open, high, low, close, volume, symbol, interval FROM _df1h"
        )
        con.unregister("_df1h")
        print(f"  Inserted {len(resampled):,} rows (1h resampled)")

    con.close()
    print(f"\nDone. Total rows: {total_rows:,}")
    print(f"Database: {db_path.resolve()}")


if __name__ == "__main__":
    main()
