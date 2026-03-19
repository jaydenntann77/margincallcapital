"""
Marimo EDA notebook for crypto OHLCV research.
Run with: marimo edit research/eda_marimo.py

Equivalent Jupyter notebook: research/Koo an's EDA.ipynb — keep both in sync.
"""

import marimo

app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    return mo, duckdb, pd, plt, mdates, np


@app.cell
def _(mo, duckdb, pd):
    # Connect to DuckDB and load available symbols
    DB_PATH = "data/research.duckdb"
    con = duckdb.connect(DB_PATH, read_only=True)

    symbols = con.execute(
        "SELECT DISTINCT symbol FROM ohlcv WHERE interval='1h' ORDER BY symbol"
    ).df()["symbol"].tolist()

    symbol_picker = mo.ui.dropdown(
        options=symbols,
        value=symbols[0] if symbols else None,
        label="Token"
    )
    interval_picker = mo.ui.dropdown(
        options=["1h"],
        value="1h",
        label="Interval"
    )
    mo.hstack([symbol_picker, interval_picker])
    return con, symbols, symbol_picker, interval_picker


@app.cell
def _(con, symbol_picker, interval_picker, pd):
    # Load OHLCV for selected symbol
    sym = symbol_picker.value
    itv = interval_picker.value

    df = con.execute(
        f"SELECT * FROM ohlcv WHERE symbol=? AND interval=? ORDER BY ts",
        [sym, itv]
    ).df()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    print(f"Loaded {len(df)} rows for {sym} ({itv}) | {df.index.min().date()} → {df.index.max().date()}")
    df
    return df, sym, itv


@app.cell
def _(df, sym, plt, pd):
    # Price + Volume chart
    fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    _ax1.plot(df.index, df["close"], linewidth=1, color="#2196F3", label="Close")
    _ax1.set_title(f"{sym} — Price & Volume", fontsize=14)
    _ax1.set_ylabel("Price (USD)")
    _ax1.legend()
    _ax1.grid(alpha=0.3)

    _ax2.bar(df.index, df["volume"], color="#90CAF9", alpha=0.7, label="Volume")
    _ax2.set_ylabel("Volume")
    _ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return fig,


@app.cell
def _(df, sym, plt, pd, np):
    # Returns distribution + stats
    returns = df["close"].pct_change().dropna()

    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram
    axes[0].hist(returns, bins=60, color="#4CAF50", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title(f"{sym} Hourly Returns Distribution")
    axes[0].set_xlabel("Return")
    axes[0].set_ylabel("Count")

    # Rolling 30d annualized vol
    rolling_vol = returns.rolling(30*24).std() * np.sqrt(24*365)
    axes[1].plot(df.index[1:], rolling_vol, color="#FF9800", linewidth=1)
    axes[1].set_title("Rolling 30d Annualized Vol")
    axes[1].set_ylabel("Volatility")
    axes[1].grid(alpha=0.3)

    # Cumulative return
    cumret = (1 + returns).cumprod()
    axes[2].plot(df.index[1:], cumret, color="#9C27B0", linewidth=1)
    axes[2].axhline(1, color="gray", linestyle="--", linewidth=0.8)
    axes[2].set_title("Cumulative Return")
    axes[2].set_ylabel("Cumulative Return (1=entry)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    # Print stats
    sharpe = (returns.mean() / returns.std()) * np.sqrt(24 * 365)
    print(f"=== {sym} Stats ===")
    print(f"Mean hourly return : {returns.mean()*100:.4f}%")
    print(f"Std hourly return  : {returns.std()*100:.4f}%")
    print(f"Annualized Sharpe  : {sharpe:.2f}")
    print(f"Max drawdown       : {((cumret / cumret.cummax()) - 1).min()*100:.2f}%")
    print(f"Skewness           : {returns.skew():.3f}")
    print(f"Kurtosis           : {returns.kurtosis():.3f}")
    plt.gca()
    return fig2, returns, cumret, sharpe


@app.cell
def _(df, sym, plt, pd, np):
    # Technical indicators: MA20, MA50, RSI14
    close = df["close"]

    # MAs
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    fig3, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})

    _ax1.plot(df.index, close, linewidth=1, color="#2196F3", label="Close", alpha=0.8)
    _ax1.plot(df.index, ma20, linewidth=1.2, color="#FF9800", label="MA20")
    _ax1.plot(df.index, ma50, linewidth=1.2, color="#F44336", label="MA50")
    _ax1.set_title(f"{sym} — Price + MA20/50 + RSI14")
    _ax1.set_ylabel("Price")
    _ax1.legend()
    _ax1.grid(alpha=0.3)

    _ax2.plot(df.index, rsi, linewidth=1, color="#9C27B0")
    _ax2.axhline(70, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    _ax2.axhline(30, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    _ax2.axhline(50, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    _ax2.set_ylim(0, 100)
    _ax2.set_ylabel("RSI14")
    _ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return fig3, ma20, ma50, rsi


@app.cell
def _(con, pd, plt, np):
    # Correlation heatmap across all symbols
    import seaborn as sns

    all_data = con.execute(
        "SELECT ts, symbol, close FROM ohlcv WHERE interval='1h' ORDER BY ts"
    ).df()
    all_data["ts"] = pd.to_datetime(all_data["ts"])

    pivot = all_data.pivot(index="ts", columns="symbol", values="close")
    returns_all = pivot.pct_change().dropna()

    # Drop BTC-USDT (duplicate of BTC-USD)
    if "BTC-USDT" in returns_all.columns:
        returns_all = returns_all.drop(columns=["BTC-USDT"])

    corr = returns_all.corr()

    fig4, _ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, ax=_ax,
                annot_kws={"size": 11})
    _ax.set_title("Hourly Returns Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.gca()
    return fig4, returns_all, corr, pivot


@app.cell
def _(returns_all, pd, plt, np):
    # Hour-of-day seasonality: avg return by UTC hour
    returns_df = returns_all.copy()
    returns_df.index = pd.to_datetime(returns_df.index, utc=True)
    returns_df["hour"] = returns_df.index.hour

    hourly_avg = returns_df.groupby("hour").mean() * 100  # in %

    fig5, _ax = plt.subplots(figsize=(12, 5))
    for _col in hourly_avg.columns:
        _ax.plot(hourly_avg.index, hourly_avg[_col], marker="o", markersize=3,
                linewidth=1, label=_col, alpha=0.7)
    _ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    _ax.set_xlabel("UTC Hour")
    _ax.set_ylabel("Avg Hourly Return (%)")
    _ax.set_title("Hour-of-Day Seasonality (UTC)")
    _ax.legend(loc="upper right", fontsize=8)
    _ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return fig5, hourly_avg


@app.cell
def _(returns_all, pd, plt, np):
    # Autocorrelation at lag 1-48 for each symbol (momentum vs mean reversion)
    fig6, _ax = plt.subplots(figsize=(12, 5))

    lags = range(1, 49)
    for _col in returns_all.columns:
        acfs = [returns_all[_col].autocorr(lag=l) for l in lags]
        _ax.plot(list(lags), acfs, linewidth=1, label=_col, alpha=0.7)

    _ax.axhline(0, color="black", linewidth=1)
    _ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    _ax.axhline(-0.05, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    _ax.set_xlabel("Lag (hours)")
    _ax.set_ylabel("Autocorrelation")
    _ax.set_title("Return Autocorrelation (lag 1–48h)\n>0 = momentum, <0 = mean reversion")
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return fig6,


if __name__ == "__main__":
    app.run()
