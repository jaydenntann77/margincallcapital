"""Utilities for loading price data and running a vectorized long/short style backtest.

How to use this module
1. Build a price matrix with timestamps as index and symbols as columns.
2. Build a target-weights matrix with the same index/columns as prices.
3. Instantiate VectorizedBacktester with prices, weights, and optional benchmark.
4. Call run(), then print_tearsheet() and plot_charts().

Expected format for prices and weights
- prices: pandas.DataFrame
    - index: DatetimeIndex (ascending, unique, timezone-consistent)
    - columns: symbol strings (for example, BTC-USD, ETH-USD)
    - values: numeric close prices
- weights: pandas.DataFrame
    - same symbol columns as prices
    - index at the same timestamps as prices (exact match is best)
    - values are portfolio target weights per symbol at each timestamp

Signal timing convention
- The engine executes with a one-bar delay:
    executed_weights = weights.shift(1)
- If your signal is computed at time t, that allocation is applied to returns from t to t+1.
- This helps avoid lookahead bias when signals are derived from bar-close data.

Additional parameters
- benchmark_prices (optional): pandas.Series of benchmark close prices.
    It is reindexed to the strategy timestamps and forward-filled.
- fee_rate: transaction cost applied to turnover each period.
    Turnover is sum(abs(delta_weight)) across assets.
- initial_capital: starting equity value used to scale the equity curve.
- periods_per_year: annualization factor used for Sharpe/Sortino/volatility.
    Examples:
    - 1 minute bars: 525600
    - 1 hour bars: 8760
    - daily bars: 365

Notes
- prices and weights are aligned with an inner join in the constructor.
- Missing returns and missing executed weights are treated as 0.0 in run().
- Weights are not normalized by this module; normalize upstream if needed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb
import shutil
import tempfile
from pathlib import Path
import scipy.stats as stats

class GetPriceData:
    """Load a pivoted close-price matrix from DuckDB.

    The query expects a table named ohlcv with at least:
    - ts (timestamp)
    - symbol (string)
    - close (numeric)
    """

    def __init__(self, db_path: str):
        """Create a data loader.

        Args:
            db_path: Path to the DuckDB file.
        """
        self.db_path = str(Path(db_path).expanduser().resolve())

    def _connect_readonly(self):
        """Open a read-only DuckDB connection with lock-safe fallback on Windows."""
        try:
            return duckdb.connect(self.db_path, read_only=True)
        except Exception as exc:
            msg = str(exc).lower()
            is_locked = (
                "being used by another process" in msg
                or "used by another process" in msg
            )
            if not is_locked:
                raise

            db_path = Path(self.db_path)
            tmp_dir = Path(tempfile.gettempdir()) / "margincallcapital"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            db_copy_path = tmp_dir / f"{db_path.stem}_copy{db_path.suffix}"

            try:
                shutil.copy2(db_path, db_copy_path)
                print(f"DuckDB file is locked; using temp copy at: {db_copy_path}")
                return duckdb.connect(str(db_copy_path), read_only=True)
            except Exception as copy_exc:
                raise RuntimeError(
                    "DuckDB file is locked and temp-copy fallback failed. "
                    f"Original error: {exc}. Copy error: {copy_exc}"
                ) from copy_exc

    def fetch_price_data(self) -> pd.DataFrame:
        """Fetch close prices as a symbol-wide matrix.

        Returns:
            pandas.DataFrame indexed by ts with symbols as columns and close
            prices as values.
        """
        # Create a lock-safe read-only connection to DuckDB.
        con = self._connect_readonly()
        
        # Prepare the SQL query to fetch data for the specified symbols and date range
        query = f"""
            SELECT ts, symbol, close
            FROM ohlcv
            WHERE symbol NOT LIKE '%USDT%'
            ORDER BY ts ASC
        """

        try:
            # Execute the query and load the data into a DataFrame
            df = con.execute(query).fetchdf()
        finally:
            con.close()

        # Pivot the DataFrame to have timestamps as index and symbols as columns (for close prices)
        close_price_df = df.pivot(index='ts', columns='symbol', values='close')
        
        return close_price_df

class VectorizedBacktester:
    """Vectorized portfolio backtester using target weights and close prices.

    Key requirement:
    - prices and weights should represent the same tradable universe over time.
    - Each row in weights is the target portfolio allocation at that timestamp.
    - Allocations are applied with a one-period lag to avoid lookahead bias.
    """

    def __init__(self, prices: pd.DataFrame, weights: pd.DataFrame, 
                 benchmark_prices: pd.Series = None,
                 fee_rate: float = 0.001, slippage: float = 0.0005, initial_capital: float = 10000.0, 
                 periods_per_year: int = 8760,
                 start_date: str = None, 
                 end_date: str = None):
        """Initialize the backtester.

        Args:
            prices: Price matrix (index=time, columns=symbol, values=close).
            weights: Target weight matrix aligned to prices. Values are desired
                allocations per symbol (for example, 0.2 means 20% capital).
            benchmark_prices: Optional benchmark close-price series.
            fee_rate: Proportional cost per unit turnover per bar.
            slippage: Proportional cost applied to gross returns to simulate market impact.
            initial_capital: Starting capital for equity curve simulation.
            periods_per_year: Annualization factor for risk metrics.
            start_date: Optional start date string (e.g., '2022-01-01') to slice data.
            end_date: Optional end date string (e.g., '2023-01-01') to slice data.
        """
        
        # 1. Slice the data by date BEFORE alignment to optimize memory
        if start_date is not None:
            prices = prices.loc[start_date:]
            weights = weights.loc[start_date:]
            if benchmark_prices is not None:
                benchmark_prices = benchmark_prices.loc[start_date:]
                
        if end_date is not None:
            prices = prices.loc[:end_date]
            weights = weights.loc[:end_date]
            if benchmark_prices is not None:
                benchmark_prices = benchmark_prices.loc[:end_date]

        # 2. Align indexes to ensure no shape mismatches
        self.prices, self.weights = prices.align(weights, join='inner')
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.periods_per_year = periods_per_year
        
        # 3. Handle the benchmark
        if benchmark_prices is not None:
            # Reindex benchmark to match the exact timestamps of our strategy
            self.benchmark_prices = benchmark_prices.reindex(self.prices.index).ffill()
        else:
            self.benchmark_prices = None
            
        self.results = None
        self.metrics = {}
        
        # Save this as a class attribute so get_holdings() can access it later
        self.executed_weights = None

    def run(self):
        """Run the strategy and optional benchmark backtest.

        Returns:
            pandas.DataFrame indexed by time with:
            - Net_Return
            - Turnover
            - Equity
            - Drawdown
            and benchmark columns when benchmark_prices is provided.
        """
        # --- 1. Strategy Calculation ---
        asset_returns = self.prices.pct_change().fillna(0.0)
        
        # CRITICAL UPDATE: Save to self so we can query holdings later!
        self.executed_weights = self.weights.shift(1).fillna(0.0)
        
        gross_returns = (self.executed_weights * asset_returns).sum(axis=1)
        turnover = self.executed_weights.diff().abs().sum(axis=1).fillna(0.0)
        tc_drag = turnover * (self.fee_rate + self.slippage)
        net_returns = gross_returns - tc_drag
        
        equity = self.initial_capital * (1 + net_returns).cumprod()
        drawdown = (equity - equity.cummax()) / equity.cummax()
        
        self.results = pd.DataFrame({
            'Net_Return': net_returns,
            'Turnover': turnover,
            'Equity': equity,
            'Drawdown': drawdown
        }, index=self.prices.index)
        
        # --- 2. Benchmark Calculation ---
        if self.benchmark_prices is not None:
            bench_returns = self.benchmark_prices.pct_change().fillna(0.0)
            bench_equity = self.initial_capital * (1 + bench_returns).cumprod()
            bench_drawdown = (bench_equity - bench_equity.cummax()) / bench_equity.cummax()
            
            self.results['Bench_Return'] = bench_returns
            self.results['Bench_Equity'] = bench_equity
            self.results['Bench_Drawdown'] = bench_drawdown
            
        self._calculate_tearsheet()
        return self.results

    def _calc_stats(self, returns: pd.Series, equity: pd.Series, drawdown: pd.Series) -> dict:
        """Calculates standardized metrics, now including PSR."""
        compounded_return = (equity.iloc[-1] / self.initial_capital) - 1
        n_periods = len(returns)
        num_years = max(n_periods / self.periods_per_year, 0.001)
        
        ann_return = (1 + compounded_return) ** (1 / num_years) - 1
        ann_vol = returns.std() * np.sqrt(self.periods_per_year)
        
        # Risk Metrics
        sharpe_ann = ann_return / ann_vol if ann_vol != 0 else 0.0
        max_dd = drawdown.min()
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0
        
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(self.periods_per_year)
        sortino = ann_return / downside_vol if downside_vol != 0 else 0.0
        
        # --- Probabilistic Sharpe Ratio (PSR) ---
        # Calculate per-period Sharpe to match the statistical moments
        sr_period = returns.mean() / returns.std() if returns.std() != 0 else 0.0
        skew = returns.skew()
        exc_kurt = returns.kurt()
        
        # Variance of the Sharpe Ratio estimate (Lopez de Prado)
        sr_var = (1 / n_periods) * (1 + 0.5 * sr_period**2 - skew * sr_period + (exc_kurt / 4) * sr_period**2)
        sr_std_err = np.sqrt(np.maximum(sr_var, 0)) # Ensure no negative variance due to float math
        
        # PSR: Probability that the true Sharpe is > 0
        if sr_std_err > 0:
            psr_z_score = (sr_period - 0.0) / sr_std_err
            psr = stats.norm.cdf(psr_z_score)
        else:
            psr = 0.0
            
        active_periods = (returns != 0).sum()
        win_rate = (returns > 0).sum() / active_periods if active_periods > 0 else 0.0

        return {
            'Total Return (%)': compounded_return * 100,
            'Ann. Return (%)': ann_return * 100,
            'Ann. Volatility (%)': ann_vol * 100,
            'Max Drawdown (%)': max_dd * 100,
            'Sharpe Ratio': sharpe_ann,
            'Probabilistic Sharpe (%)': psr * 100,
            'Sortino Ratio': sortino,
            'Calmar Ratio': calmar,
            'Win Rate (%)': win_rate * 100
        }

    def _calculate_tearsheet(self):
        """Compiles metrics for the Strategy and Benchmark."""
        if self.results is None:
            raise ValueError("Run the backtest first!")
            
        self.metrics['Strategy'] = self._calc_stats(
            self.results['Net_Return'], self.results['Equity'], self.results['Drawdown']
        )
        
        # Add strategy-specific stats
        avg_turnover = self.results['Turnover'].mean() * 100 * (self.periods_per_year / 365)
        self.metrics['Strategy']['Daily Turnover (%)'] = avg_turnover
        
        if self.benchmark_prices is not None:
            self.metrics['Benchmark'] = self._calc_stats(
                self.results['Bench_Return'], self.results['Bench_Equity'], self.results['Bench_Drawdown']
            )
            self.metrics['Benchmark']['Daily Turnover (%)'] = 0.0 # Benchmark is Buy & Hold

    def print_tearsheet(self):
        """Prints a side-by-side comparison."""
        print("="*60)
        print(f"{'STRATEGY TEARSHEET':^60}")
        print("="*60)
        
        has_bench = 'Benchmark' in self.metrics
        header = f"{'Metric':<25} | {'Strategy':>12}"
        if has_bench: header += f" | {'Benchmark':>12}"
        
        print(header)
        print("-" * len(header))
        
        # Use Strategy keys as the master list
        for key in self.metrics['Strategy'].keys():
            strat_val = self.metrics['Strategy'][key]
            row = f"{key:<25} | {strat_val:>12.2f}"
            
            if has_bench:
                bench_val = self.metrics['Benchmark'].get(key, 0.0)
                row += f" | {bench_val:>12.2f}"
                
            print(row)
        print("="*60)

    def plot_charts(self, rolling_window_days=30):
        """Plots Strategy vs Benchmark equity, drawdown, and Rolling Sharpe with PSR CI."""
        # Convert days to periods for the rolling window
        rolling_n = int((self.periods_per_year / 365) * rolling_window_days)
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1.5]}, sharex=True)
        
        # --- Plot 1: Equity Curves ---
        axes[0].plot(self.results.index, self.results['Equity'], color='blue', linewidth=2, label='Strategy (Net)')
        if self.benchmark_prices is not None:
            axes[0].plot(self.results.index, self.results['Bench_Equity'], color='orange', linewidth=2, alpha=0.8, label='Benchmark (BTC)')
        axes[0].set_title('Portfolio Equity Curve')
        axes[0].set_ylabel('Capital ($)')
        axes[0].set_yscale('log')
        axes[0].legend(loc='upper left')
        axes[0].grid(alpha=0.3)
        
        # --- Plot 2: Drawdown Curves ---
        axes[1].fill_between(self.results.index, self.results['Drawdown'] * 100, 0, color='blue', alpha=0.2)
        axes[1].plot(self.results.index, self.results['Drawdown'] * 100, color='blue', linewidth=1, label=f'{rolling_window_days}-Day Strategy DD')
        if self.benchmark_prices is not None:
            axes[1].plot(self.results.index, self.results['Bench_Drawdown'] * 100, color='orange', linewidth=1, alpha=0.8, label=f'{rolling_window_days}-Day Benchmark DD')
        axes[1].set_title('Drawdown (%)')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].legend(loc='lower left')
        axes[1].grid(alpha=0.3)
        
        # --- Plot 3: Rolling Sharpe Ratio with PSR Confidence Intervals ---
        returns = self.results['Net_Return']
        
        # Rolling stats
        roll_mean = returns.rolling(rolling_n).mean()
        roll_std = returns.rolling(rolling_n).std()
        roll_skew = returns.rolling(rolling_n).skew()
        roll_exc_kurt = returns.rolling(rolling_n).kurt()
        
        # Rolling Per-period Sharpe
        roll_sr_period = roll_mean / roll_std
        
        # Rolling PSR Standard Error Formula
        roll_sr_var = (1 / rolling_n) * (
            1 + 0.5 * roll_sr_period**2 - roll_skew * roll_sr_period + (roll_exc_kurt / 4) * roll_sr_period**2
        )
        roll_sr_se = np.sqrt(np.maximum(roll_sr_var, 0))
        
        # Annualize everything for the plot
        ann_factor = np.sqrt(self.periods_per_year)
        roll_sr_ann = roll_sr_period * ann_factor
        roll_se_ann = roll_sr_se * ann_factor
        
        # 95% Confidence Intervals (1.96 standard deviations)
        ci_upper = roll_sr_ann + (1.96 * roll_se_ann)
        ci_lower = roll_sr_ann - (1.96 * roll_se_ann)
        
        axes[2].plot(roll_sr_ann.index, roll_sr_ann, color='purple', linewidth=1.5, label=f'{rolling_window_days}-Day Rolling Sharpe')
        axes[2].fill_between(roll_sr_ann.index, ci_lower, ci_upper, color='purple', alpha=0.15, label='95% PSR Confidence Interval')
        axes[2].axhline(0.0, color='black', linewidth=1, alpha=0.6)
        
        axes[2].set_title(f'Rolling Annualized Sharpe Ratio (Adjusted for Non-Normal Returns)')
        axes[2].set_ylabel('Ann. Sharpe')
        axes[2].legend(loc='best')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def get_holdings(self, timestamp: str) -> dict:
        """
        Retrieves the exact portfolio allocation for a given timestamp.
        Example: engine.get_holdings('2024-03-05 14:00:00')

        Timezone handling:
        - If the backtest index is timezone-aware, naive inputs are localized to
          that timezone and aware inputs are converted.
        - If the backtest index is timezone-naive, aware inputs are converted to
          naive UTC for comparison.
        """
        if self.executed_weights is None:
            raise ValueError("You must run() the backtest before checking holdings.")

        # Normalize timestamp to the same timezone representation as the index.
        index = self.executed_weights.index
        ts = pd.Timestamp(timestamp)

        if isinstance(index, pd.DatetimeIndex):
            if index.tz is not None:
                if ts.tzinfo is None:
                    ts = ts.tz_localize(index.tz)
                else:
                    ts = ts.tz_convert(index.tz)
            elif ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)

        # Ensure the timestamp exists in our backtest index.
        if ts not in index:
            # Fallback: get the closest preceding timestamp
            idx = index.get_indexer([ts], method='pad')[0]
            if idx == -1:
                return {"Error": "Timestamp is before the backtest started."}
            ts = index[idx]
            print(f"Exact timestamp not found. Showing holdings for closest prior bar: {ts}")
            
        # Extract the row for this timestamp
        row = self.executed_weights.loc[ts]
        
        # Filter for coins where weight > 0 and convert to dictionary
        active_holdings = row[row > 0].to_dict()
        
        return active_holdings
    
    def get_holdings_history(self, output_format: str = 'long') -> pd.DataFrame:
        """
        Returns a complete history of all portfolio members over time.
        
        Args:
            output_format: 
                'long' -> Standard database format (Timestamp | Symbol | Weight). Best for analysis.
                'wide' -> Visual format (Timestamp | Active_Members). Best for quick reading.
        """
        if self.executed_weights is None:
            raise ValueError("You must run() the backtest before checking holdings.")
            
        if output_format == 'long':
            # 1. Prepare the DataFrame for melting
            df = self.executed_weights.copy()
            df.index.name = 'Timestamp'
            
            # 2. Melt the wide matrix into a long format
            # This turns every (Time, Symbol) coordinate into its own row
            long_df = df.reset_index().melt(
                id_vars='Timestamp', 
                var_name='Symbol', 
                value_name='Weight'
            )
            
            # 3. Filter out zero weights (things we don't hold) and sort
            active_holdings = long_df[long_df['Weight'] > 0].copy()
            active_holdings.sort_values(by=['Timestamp', 'Weight'], ascending=[True, False], inplace=True)
            
            return active_holdings.set_index('Timestamp')
            
        elif output_format == 'wide':
            # Returns a single column containing a dictionary of {Symbol: Weight} for each bar
            def get_active_members(row):
                active = row[row > 0]
                if active.empty:
                    return "CASH"
                # Format nicely: "BTC: 20.0%, ETH: 20.0%"
                return ", ".join([f"{sym}: {w*100:.1f}%" for sym, w in active.items()])
                
            members = self.executed_weights.apply(get_active_members, axis=1)
            return pd.DataFrame({'Portfolio_Members': members})
            
        else:
            raise ValueError("output_format must be 'long' or 'wide'.")