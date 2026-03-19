"""
Binance Vision spot klines: monthly zips + daily zips fallback for the current month.

Why daily fallback: monthly `*-1m-YYYY-MM.zip` often does not exist for the *in-progress*
UTC month, so series used to end around the last day of the previous month unless you
pull `data/spot/daily/klines/...` for each day.
"""

from __future__ import annotations

import calendar
import io
import zipfile
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

BINANCE_VISION = "https://data.binance.vision/data/spot"
USER_AGENT = {"User-Agent": "margincallcapital-research/1.0"}

KLINES_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_vol",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def parse_timestamp(ts_val: int) -> datetime:
    if ts_val > 1e15:
        ts_val = ts_val // 1000
    return datetime.fromtimestamp(ts_val / 1000.0, tz=timezone.utc)


def month_range_calendar(months_back: int) -> list[tuple[int, int]]:
    """
    Oldest-first (year, month) tuples for the last `months_back` **calendar** months,
    **including** the current UTC month (fixes “data stops at 1st of month” vs 30-day heuristic).
    """
    today = datetime.now(timezone.utc)
    y, m = today.year, today.month
    out: list[tuple[int, int]] = []
    for _ in range(months_back):
        out.insert(0, (y, m))
        if m == 1:
            y, m = y - 1, 12
        else:
            m -= 1
    return out


def _dataframe_from_zip(content: bytes) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            names = z.namelist()
            if not names:
                return None
            with z.open(names[0]) as f:
                df = pd.read_csv(f, header=None, names=KLINES_COLS)
    except (zipfile.BadZipFile, OSError):
        return None
    if df.empty:
        return None
    df["ts"] = df["open_time"].apply(parse_timestamp)
    df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df


def download_binance_monthly_klines(
    symbol: str, year: int, month: int, interval: str
) -> pd.DataFrame | None:
    ym = f"{year}-{month:02d}"
    fname = f"{symbol}-{interval}-{ym}.zip"
    url = f"{BINANCE_VISION}/monthly/klines/{symbol}/{interval}/{fname}"
    try:
        r = requests.get(url, timeout=90, headers=USER_AGENT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
    except requests.RequestException:
        return None
    return _dataframe_from_zip(r.content)


def download_binance_daily_klines_month(
    symbol: str, year: int, month: int, interval: str
) -> pd.DataFrame | None:
    """Concatenate daily Vision zips for each day in month through today (UTC)."""
    today = datetime.now(timezone.utc).date()
    last_dom = calendar.monthrange(year, month)[1]
    end_d = min(today, date(year, month, last_dom))
    start_d = date(year, month, 1)
    parts: list[pd.DataFrame] = []
    d = start_d
    while d <= end_d:
        ds = d.isoformat()
        fname = f"{symbol}-{interval}-{ds}.zip"
        url = f"{BINANCE_VISION}/daily/klines/{symbol}/{interval}/{fname}"
        try:
            r = requests.get(url, timeout=90, headers=USER_AGENT)
            if r.status_code == 404:
                d += timedelta(days=1)
                continue
            r.raise_for_status()
            part = _dataframe_from_zip(r.content)
            if part is not None and not part.empty:
                parts.append(part)
        except requests.RequestException:
            pass
        d += timedelta(days=1)
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(subset=["ts"]).sort_values("ts")


def download_binance_klines_calendar_month(
    symbol: str, year: int, month: int, interval: str, *, verbose_label: str = ""
) -> pd.DataFrame | None:
    """
    Prefer monthly zip; if missing, use daily zips for that calendar month (covers current month).
    """
    df = download_binance_monthly_klines(symbol, year, month, interval)
    if df is not None and not df.empty:
        return df
    df_d = download_binance_daily_klines_month(symbol, year, month, interval)
    if df_d is not None and not df_d.empty and verbose_label:
        print(f"    {verbose_label} {year}-{month:02d}: monthly missing — used daily zips ({len(df_d):,} rows)")
    return df_d
