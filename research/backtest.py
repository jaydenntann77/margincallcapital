"""
Simple momentum backtest — run directly with Python.
Sphinx AI can edit the signal logic in generate_signals() to test new ideas.

Usage:
    python research/backtest.py
    python research/backtest.py --symbol ETH-USD --fast 10 --slow 30
"""

import duckdb
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "research.duckdb"
# Use research_1m.duckdb for 1m+1h Roostoo data: --db data/research_1m.duckdb


# ── Data loading ─────────────────────────────────────────────────────────────

def load_ohlcv(symbol: str = "BTC-USD", interval: str = "1h") -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT ts, open, high, low, close, volume FROM ohlcv "
        "WHERE symbol=? AND interval=? ORDER BY ts",
        [symbol, interval]
    ).df()
    con.close()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    return df


# ── Signal generation (EDIT THIS to test different strategies) ────────────────

def generate_signals(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """
    MA crossover momentum signal.
    Returns 1 (long), 0 (flat) — no short, no leverage.

    Sphinx AI: replace this function body to test different signals.
    Ideas:
      - RSI mean reversion: buy when RSI < 35, sell when RSI > 65
      - Breakout: buy when close > rolling_max(24), sell when close < ma20
      - Volume-confirmed: only buy when volume > 2x rolling_mean(volume, 20)
    """
    df = df.copy()

    # Moving averages
    df["ma_fast"] = df["close"].rolling(fast).mean()
    df["ma_slow"] = df["close"].rolling(slow).mean()

    # RSI14
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Signal: long only when fast MA > slow MA AND RSI < 78 (not overbought)
    df["signal"] = 0
    df.loc[(df["ma_fast"] > df["ma_slow"]) & (df["rsi"] < 78), "signal"] = 1

    return df


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, initial_capital: float = 50_000.0) -> dict:
    """
    Vectorized backtest — long-only, no leverage, no shorting.
    Position = signal × capital. No partial fills, no slippage.
    """
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["strategy_returns"] = df["signal"].shift(1) * df["returns"]
    df["equity"] = initial_capital * (1 + df["strategy_returns"].fillna(0)).cumprod()

    # Metrics
    strat_ret = df["strategy_returns"].dropna()
    buy_hold_ret = df["returns"].dropna()

    total_return = df["equity"].iloc[-1] / initial_capital - 1
    bh_total = (df["close"].iloc[-1] / df["close"].iloc[0]) - 1

    sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(24 * 365) if strat_ret.std() > 0 else 0
    bh_sharpe = (buy_hold_ret.mean() / buy_hold_ret.std()) * np.sqrt(24 * 365) if buy_hold_ret.std() > 0 else 0

    rolling_max = df["equity"].cummax()
    drawdown = (df["equity"] - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Trade count (entries only)
    entries = (df["signal"].diff() == 1).sum()

    # Win rate (close each "trade" when signal flips off)
    signal_changes = df["signal"].diff().fillna(0)
    entry_dates = df.index[signal_changes == 1]
    exit_dates = df.index[signal_changes == -1]

    wins = 0
    trade_returns = []
    for entry in entry_dates:
        exits_after = exit_dates[exit_dates > entry]
        if len(exits_after) == 0:
            exit_date = df.index[-1]
        else:
            exit_date = exits_after[0]
        trade_ret = df.loc[exit_date, "close"] / df.loc[entry, "close"] - 1
        trade_returns.append(trade_ret)
        if trade_ret > 0:
            wins += 1

    win_rate = wins / len(trade_returns) if trade_returns else 0
    avg_trade = np.mean(trade_returns) if trade_returns else 0

    return {
        "total_return_pct": total_return * 100,
        "bh_return_pct": bh_total * 100,
        "sharpe": round(sharpe, 3),
        "bh_sharpe": round(bh_sharpe, 3),
        "max_drawdown_pct": max_dd * 100,
        "num_trades": entries,
        "win_rate_pct": win_rate * 100,
        "avg_trade_pct": avg_trade * 100,
        "final_equity": df["equity"].iloc[-1],
        "equity_series": df["equity"],
        "signal_series": df["signal"],
    }


# ── Print + plot ──────────────────────────────────────────────────────────────

def print_results(symbol: str, result: dict):
    print(f"\n{'='*50}")
    print(f"  Backtest: {symbol}")
    print(f"{'='*50}")
    print(f"  Strategy return : {result['total_return_pct']:+.2f}%")
    print(f"  Buy & hold      : {result['bh_return_pct']:+.2f}%")
    print(f"  Sharpe (strat)  : {result['sharpe']:.3f}")
    print(f"  Sharpe (B&H)    : {result['bh_sharpe']:.3f}")
    print(f"  Max drawdown    : {result['max_drawdown_pct']:.2f}%")
    print(f"  Trades          : {result['num_trades']}")
    print(f"  Win rate        : {result['win_rate_pct']:.1f}%")
    print(f"  Avg trade       : {result['avg_trade_pct']:+.3f}%")
    print(f"  Final equity    : ${result['final_equity']:,.0f}")
    print(f"{'='*50}\n")


def plot_equity(symbol: str, result: dict):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(result["equity_series"], label="Strategy equity", color="#2196F3")
    ax1.set_title(f"{symbol} — Momentum Backtest (MA crossover + RSI gate)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.fill_between(result["signal_series"].index,
                     result["signal_series"], 0,
                     alpha=0.4, color="#4CAF50", label="In market")
    ax2.set_ylabel("In market")
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"data/{symbol.replace('/', '-')}_backtest.png", dpi=120, bbox_inches="tight")
    print(f"  Chart saved → data/{symbol.replace('/', '-')}_backtest.png")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run momentum backtest")
    parser.add_argument("--symbol", default="BTC-USD")
    parser.add_argument("--fast", type=int, default=20, help="Fast MA period")
    parser.add_argument("--slow", type=int, default=50, help="Slow MA period")
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--all", action="store_true", help="Run for all symbols")
    parser.add_argument("--db", default=None, help="DuckDB path (default: data/research.duckdb)")
    args = parser.parse_args()

    if args.db:
        DB_PATH = Path(args.db)

    if args.all:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        syms = con.execute(
            "SELECT DISTINCT symbol FROM ohlcv WHERE interval='1h' AND symbol != 'BTC-USDT'"
        ).df()["symbol"].tolist()
        con.close()
        for sym in syms:
            try:
                df = load_ohlcv(sym)
                df = generate_signals(df, args.fast, args.slow)
                result = run_backtest(df, args.capital)
                print_results(sym, result)
            except Exception as e:
                print(f"  {sym}: ERROR — {e}")
    else:
        df = load_ohlcv(args.symbol)
        df = generate_signals(df, args.fast, args.slow)
        result = run_backtest(df, args.capital)
        print_results(args.symbol, result)
        plot_equity(args.symbol, result)
