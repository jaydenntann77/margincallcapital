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
import duckdb
import shutil
import tempfile
from pathlib import Path
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        """Initialize the backtester."""
        
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
        self.pnl_by_asset = None
        self.stress_summary = None
        self.stress_test_results = None
        self._stress_slices = {}

    def run(self):
        """Run the strategy and optional benchmark backtest.

        Returns:
            pandas.DataFrame indexed by time with metrics.
        """
        # --- 1. Strategy Calculation ---
        asset_returns = self.prices.pct_change().fillna(0.0)
        
        # Save to self so we can query holdings later
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
        self._calculate_pnl_attribution()
        return self.results

    def _calculate_pnl_attribution(self):
        """Calculates net dollar PnL contribution per traded asset."""
        asset_returns = self.prices.pct_change().fillna(0.0)
        asset_turnover = self.executed_weights.diff().abs().fillna(0.0)
        asset_tc_drag = asset_turnover * (self.fee_rate + self.slippage)
        
        net_asset_returns = (self.executed_weights * asset_returns) - asset_tc_drag
        equity_shifted = self.results['Equity'].shift(1).fillna(self.initial_capital)
        asset_dollar_pnl = net_asset_returns.multiply(equity_shifted, axis=0)
        
        # Filter for assets that actually traded during the backtest
        total_turnover = asset_turnover.sum()
        traded_assets = total_turnover[total_turnover > 0].index
        
        total_pnl = asset_dollar_pnl[traded_assets].sum()
        self.pnl_by_asset = total_pnl[total_pnl != 0].sort_values()

    def _calc_stats(self, returns: pd.Series, equity: pd.Series, drawdown: pd.Series) -> dict:
        """Calculates standardized metrics, including PSR."""
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
        sr_period = returns.mean() / returns.std() if returns.std() != 0 else 0.0
        skew = returns.skew()
        exc_kurt = returns.kurt()
        
        sr_var = (1 / n_periods) * (1 + 0.5 * sr_period**2 - skew * sr_period + (exc_kurt / 4) * sr_period**2)
        sr_std_err = np.sqrt(np.maximum(sr_var, 0)) 
        
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
            self.metrics['Benchmark']['Daily Turnover (%)'] = 0.0

    def print_tearsheet(self):
        """Prints a side-by-side comparison and PnL Summary."""
        print("="*60)
        print(f"{'STRATEGY TEARSHEET':^60}")
        print("="*60)
        
        has_bench = 'Benchmark' in self.metrics
        header = f"{'Metric':<25} | {'Strategy':>12}"
        if has_bench: header += f" | {'Benchmark':>12}"
        
        print(header)
        print("-" * len(header))
        
        for key in self.metrics['Strategy'].keys():
            strat_val = self.metrics['Strategy'][key]
            row = f"{key:<25} | {strat_val:>12.2f}"
            
            if has_bench:
                bench_val = self.metrics['Benchmark'].get(key, 0.0)
                row += f" | {bench_val:>12.2f}"
                
            print(row)
            
        print("\n" + "="*60)
        print(f"{'NET PNL CONTRIBUTION BY ASSET ($)':^60}")
        print("="*60)
        if self.pnl_by_asset is not None and not self.pnl_by_asset.empty:
            for asset, pnl in self.pnl_by_asset.items():
                print(f"{asset:<25} | ${pnl:>14,.2f}")
        else:
            print("No PnL generated.")
        print("="*60)

    def _default_stress_scenarios(self):
        """Default historical stress windows used when scenarios are not provided."""
        return {
            'Covid-19 Liquidity Crisis (Mar 2020)': ('2020-02-15', '2020-04-15'),
            'Terra/Luna Preamble (Apr 2022)': ('2022-04-01', '2022-06-01'),
            'Yen Carry Unwind (Jul 2024)': ('2024-07-15', '2024-08-30'),
            'Tariff Shock (Apr 2025)': ('2025-03-25', '2025-05-15')
        }

    def _normalize_stress_scenarios(self, scenarios):
        """Validate and normalize scenario input into ordered timestamp windows."""
        if scenarios is None:
            scenarios = self._default_stress_scenarios()

        if not isinstance(scenarios, dict) or len(scenarios) == 0:
            raise ValueError("scenarios must be a non-empty dict of {name: (start, end)}")

        normalized = []
        for name, window in scenarios.items():
            if not isinstance(window, (tuple, list)) or len(window) != 2:
                raise ValueError(f"Scenario '{name}' must be a (start_date, end_date) tuple")

            start_ts = pd.Timestamp(window[0])
            end_ts = pd.Timestamp(window[1])
            if start_ts > end_ts:
                start_ts, end_ts = end_ts, start_ts

            normalized.append((str(name), start_ts, end_ts))

        normalized.sort(key=lambda x: x[1])
        return normalized

    def _align_ts_to_results_index(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Align scenario timestamps to results index timezone semantics."""
        index = self.results.index
        if not isinstance(index, pd.DatetimeIndex):
            return ts

        if index.tz is not None:
            if ts.tzinfo is None:
                return ts.tz_localize(index.tz)
            return ts.tz_convert(index.tz)

        if ts.tzinfo is not None:
            return ts.tz_convert("UTC").tz_localize(None)
        return ts

    def _calc_window_sharpe(self, returns: pd.Series) -> float:
        """Annualized Sharpe for a pre-sliced return series."""
        std = returns.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return (returns.mean() / std) * np.sqrt(self.periods_per_year)

    def _calc_stress_window_metrics(self, name: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, slice_df: pd.DataFrame, overlap_status: str) -> dict:
        """Compute stress diagnostics for one scenario window."""
        strat_ret = slice_df['Net_Return']
        strat_eq = (1.0 + strat_ret).cumprod()
        strat_dd = (strat_eq - strat_eq.cummax()) / strat_eq.cummax()

        strat_total_ret = (strat_eq.iloc[-1] - 1.0) * 100
        strat_max_dd = strat_dd.min() * 100
        strat_vol = strat_ret.std() * np.sqrt(self.periods_per_year) * 100
        strat_sharpe = self._calc_window_sharpe(strat_ret)
        worst_day = strat_ret.min() * 100
        avg_turnover = slice_df['Turnover'].mean() * 100 * (self.periods_per_year / 365)
        cost_drag = (slice_df['Turnover'] * (self.fee_rate + self.slippage)).sum() * 100

        row = {
            'Scenario': name,
            'Requested_Start': start_ts.date().isoformat(),
            'Requested_End': end_ts.date().isoformat(),
            'Window_Start': slice_df.index[0].date().isoformat(),
            'Window_End': slice_df.index[-1].date().isoformat(),
            'Bars': int(len(slice_df)),
            'Overlap': overlap_status,
            'Strat Return %': strat_total_ret,
            'Strat Max DD %': strat_max_dd,
            'Strat Vol %': 0.0 if np.isnan(strat_vol) else strat_vol,
            'Strat Sharpe': strat_sharpe,
            'Worst Day %': worst_day,
            'Avg Daily Turnover %': avg_turnover,
            'Cost Drag %': cost_drag,
            'Bench Return %': np.nan,
            'Bench Max DD %': np.nan,
            'Bench Sharpe': np.nan,
            'Relative Return %': np.nan,
            'Warning': ''
        }

        has_bench_cols = all(col in slice_df.columns for col in ['Bench_Return', 'Bench_Equity', 'Bench_Drawdown'])
        if has_bench_cols:
            bench_ret = slice_df['Bench_Return']
            bench_eq = (1.0 + bench_ret).cumprod()
            bench_dd = (bench_eq - bench_eq.cummax()) / bench_eq.cummax()
            bench_total_ret = (bench_eq.iloc[-1] - 1.0) * 100
            bench_max_dd = bench_dd.min() * 100
            bench_sharpe = self._calc_window_sharpe(bench_ret)

            row['Bench Return %'] = bench_total_ret
            row['Bench Max DD %'] = bench_max_dd
            row['Bench Sharpe'] = bench_sharpe
            row['Relative Return %'] = strat_total_ret - bench_total_ret

        return row

    def run_stress_test(self, scenarios=None, min_bars: int = 5) -> pd.DataFrame:
        """Run historical window stress scenarios and return a summary table."""
        if self.results is None:
            raise ValueError("Run the backtest first!")

        required_cols = {'Net_Return', 'Turnover', 'Equity', 'Drawdown'}
        missing = required_cols.difference(self.results.columns)
        if missing:
            raise ValueError(f"results is missing required columns: {sorted(missing)}")

        scenario_windows = self._normalize_stress_scenarios(scenarios)
        summary_rows = []
        stress_slices = {}
        index_min = self.results.index.min()
        index_max = self.results.index.max()

        for name, start_ts, end_ts in scenario_windows:
            start_ts = self._align_ts_to_results_index(start_ts)
            end_ts = self._align_ts_to_results_index(end_ts)
            slice_df = self.results.loc[start_ts:end_ts].copy()
            if slice_df.empty or len(slice_df) < min_bars:
                continue

            overlap_status = 'full'
            if start_ts < index_min or end_ts > index_max:
                overlap_status = 'partial'

            row = self._calc_stress_window_metrics(name, start_ts, end_ts, slice_df, overlap_status)
            if overlap_status == 'partial':
                row['Warning'] = 'Requested window partially overlaps with available backtest data.'

            summary_rows.append(row)
            stress_slices[name] = slice_df

        if len(summary_rows) == 0:
            self.stress_summary = pd.DataFrame(columns=[
                'Scenario', 'Requested_Start', 'Requested_End', 'Window_Start', 'Window_End', 'Bars',
                'Overlap', 'Strat Return %', 'Strat Max DD %', 'Strat Vol %', 'Strat Sharpe',
                'Worst Day %', 'Avg Daily Turnover %', 'Cost Drag %', 'Bench Return %',
                'Bench Max DD %', 'Bench Sharpe', 'Relative Return %', 'Warning'
            ])
            self.stress_test_results = self.stress_summary
            self._stress_slices = {}
            return self.stress_summary

        summary_df = pd.DataFrame(summary_rows)
        summary_df.sort_values(by=['Window_Start', 'Scenario'], inplace=True)
        summary_df.reset_index(drop=True, inplace=True)

        self.stress_summary = summary_df
        self.stress_test_results = summary_df
        self._stress_slices = stress_slices
        return summary_df

    def print_stress_tearsheet(self, scenarios=None, min_bars: int = 5, sort_by: str = 'Window_Start'):
        """Print a tabular summary for scenario stress diagnostics."""
        if isinstance(scenarios, pd.DataFrame):
            summary = scenarios.copy()
        else:
            summary = self.run_stress_test(scenarios=scenarios, min_bars=min_bars)
        if summary.empty:
            print("No backtest data overlaps with the specified stress scenarios.")
            return summary

        print("=" * 118)
        print(f"{'SCENARIO STRESS TEARSHEET':^118}")
        print("=" * 118)

        if sort_by in summary.columns:
            summary = summary.sort_values(by=sort_by).reset_index(drop=True)

        has_bench = summary['Bench Return %'].notna().any()
        if has_bench:
            header = (
                f"{'Scenario':<34} | {'Str Ret %':>8} | {'Str DD %':>8} | {'Sharpe':>7} | "
                f"{'Bench Ret %':>10} | {'Bench DD %':>10} | {'Rel %':>8} | {'Warning':<22}"
            )
        else:
            header = (
                f"{'Scenario':<34} | {'Str Ret %':>8} | {'Str DD %':>8} | {'Sharpe':>7} | "
                f"{'Worst Day %':>10} | {'Warning':<22}"
            )

        print(header)
        print("-" * len(header))

        for _, row in summary.iterrows():
            warning = row['Warning'] if isinstance(row['Warning'], str) else ''
            if has_bench:
                line = (
                    f"{row['Scenario']:<34} | {row['Strat Return %']:>8.2f} | {row['Strat Max DD %']:>8.2f} | "
                    f"{row['Strat Sharpe']:>7.2f} | {row['Bench Return %']:>10.2f} | {row['Bench Max DD %']:>10.2f} | "
                    f"{row['Relative Return %']:>8.2f} | {warning:<22}"
                )
            else:
                line = (
                    f"{row['Scenario']:<34} | {row['Strat Return %']:>8.2f} | {row['Strat Max DD %']:>8.2f} | "
                    f"{row['Strat Sharpe']:>7.2f} | {row['Worst Day %']:>10.2f} | {warning:<22}"
                )
            print(line)

        print("=" * 118)
        return summary

    def plot_stress_test(self, scenarios=None, min_bars: int = 5):
        """Generate isolated rebased-equity stress charts for historical windows."""
        if isinstance(scenarios, pd.DataFrame):
            summary = scenarios.copy()
        else:
            summary = self.run_stress_test(scenarios=scenarios, min_bars=min_bars)
        if summary.empty:
            print("No backtest data overlaps with the specified stress scenarios.")
            return

        # Ensure slices are available when caller passes a precomputed summary frame.
        if not self._stress_slices:
            self.run_stress_test(min_bars=min_bars)

        n_rows = len(summary)
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,
            subplot_titles=summary['Scenario'].tolist()
        )

        for i, (_, row) in enumerate(summary.iterrows()):
            scenario_name = row['Scenario']
            slice_df = self._stress_slices[scenario_name]
            row_idx = i + 1

            strat_eq_rebased = (1.0 + slice_df['Net_Return']).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=strat_eq_rebased.index,
                    y=strat_eq_rebased,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Strategy',
                    legendgroup='stress_strategy',
                    showlegend=(i == 0),
                    hovertemplate="<b>Date:</b> %{x}<br><b>Rebased Eq:</b> %{y:.3f}<extra></extra>"
                ),
                row=row_idx,
                col=1,
            )

            bench_text = ""
            has_bench = all(c in slice_df.columns for c in ['Bench_Return', 'Bench_Equity', 'Bench_Drawdown'])
            if has_bench:
                bench_eq_rebased = (1.0 + slice_df['Bench_Return']).cumprod()
                fig.add_trace(
                    go.Scatter(
                        x=bench_eq_rebased.index,
                        y=bench_eq_rebased,
                        mode='lines',
                        line=dict(color='orange', width=2, dash='dot'),
                        name='Benchmark',
                        legendgroup='stress_benchmark',
                        showlegend=(i == 0),
                        hovertemplate="<b>Date:</b> %{x}<br><b>Bench Rebased Eq:</b> %{y:.3f}<extra></extra>"
                    ),
                    row=row_idx,
                    col=1,
                )
                bench_text = (
                    f"<br>Bench Ret: {row['Bench Return %']:.1f}% | "
                    f"Bench Max DD: {row['Bench Max DD %']:.1f}%"
                )

            annotation_text = (
                f"<b>Window Performance</b><br>"
                f"Strat Ret: {row['Strat Return %']:.1f}% | "
                f"Strat Max DD: {row['Strat Max DD %']:.1f}% | "
                f"Worst Day: {row['Worst Day %']:.1f}%"
                f"{bench_text}"
            )
            if isinstance(row['Warning'], str) and row['Warning']:
                annotation_text += f"<br><i>{row['Warning']}</i>"

            xref = "x domain" if row_idx == 1 else f"x{row_idx} domain"
            yref = "y domain" if row_idx == 1 else f"y{row_idx} domain"
            fig.add_annotation(
                x=0.01,
                y=0.06,
                xref=xref,
                yref=yref,
                text=annotation_text,
                showarrow=False,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.82)",
                bordercolor="rgba(0,0,0,0.25)",
                borderwidth=1,
                font=dict(size=11),
            )

            fig.add_hline(
                y=1.0,
                line_width=1,
                line_dash="solid",
                line_color="black",
                opacity=0.4,
                row=row_idx,
                col=1,
            )
            fig.update_yaxes(title_text="Rebased Eq", row=row_idx, col=1)

        fig.update_layout(
            height=max(360, 300 * n_rows),
            title_text="Historical Scenario Stress Testing (Rebased Equity)",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(t=120, b=50, l=80, r=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
            ),
        )

        fig.show()

    def plot_stress_test_charts(self, stress_test_results=None, min_bars: int = 5):
        """Backward-compatible alias for plotting stress scenarios."""
        if isinstance(stress_test_results, pd.DataFrame):
            return self.plot_stress_test(scenarios=stress_test_results, min_bars=min_bars)
        return self.plot_stress_test(min_bars=min_bars)

    def plot_charts(self, rolling_window_days=30):
        """Plots 4-pane layout: Equity, DD, Rolling Sharpe, and Asset PnL."""
        if self.results is None:
            raise ValueError("Run the backtest first!")

        rolling_n = int((self.periods_per_year / 365) * rolling_window_days)
        
        holdings_df = self.get_holdings_history(output_format='wide')
        hover_holdings = holdings_df['Portfolio_Members'].tolist()

        # Create subplots without globally shared x-axes to accommodate the bar chart
        fig = make_subplots(
            rows=4, cols=1, 
            vertical_spacing=0.06,
            subplot_titles=(
                'Portfolio Equity Curve', 
                'Drawdown (%)', 
                f'Rolling {rolling_window_days}-Day Sharpe Ratio (PSR Adjusted)',
                'Net PnL Contribution by Asset (Top and Bottom 10 Only)'
            ),
            row_heights=[0.35, 0.15, 0.20, 0.30]
        )

        # Manually link the x-axes for the first three time-series plots
        fig.update_xaxes(matches='x', row=1, col=1)
        fig.update_xaxes(matches='x', row=2, col=1)
        fig.update_xaxes(matches='x', row=3, col=1)

        # --- Plot 1: Equity Curves ---
        fig.add_trace(
            go.Scatter(
                x=self.results.index, y=self.results['Equity'],
                mode='lines', name='Strategy (Net)',
                legend='legend',
                line=dict(color='blue', width=2),
                customdata=hover_holdings,
                hovertemplate="<b>Date:</b> %{x}<br><b>Equity:</b> $%{y:,.2f}<br><b>Holdings:</b> %{customdata}<extra></extra>"
            ), row=1, col=1
        )

        if self.benchmark_prices is not None:
            fig.add_trace(
                go.Scatter(
                    x=self.results.index, y=self.results['Bench_Equity'],
                    mode='lines', name='Benchmark',
                    legend='legend',
                    line=dict(color='orange', width=2),
                    hovertemplate="<b>Date:</b> %{x}<br><b>Bench Equity:</b> $%{y:,.2f}<extra></extra>"
                ), row=1, col=1
            )
            
        fig.update_yaxes(type="log", title_text="Capital ($)", row=1, col=1)

        # --- Plot 2: Drawdown Curves ---
        fig.add_trace(
            go.Scatter(
                x=self.results.index, y=self.results['Drawdown'] * 100,
                mode='lines', name='Strategy DD',
                legend='legend2',
                line=dict(color='blue', width=1),
                fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.1)',
                hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>"
            ), row=2, col=1
        )

        if self.benchmark_prices is not None:
            fig.add_trace(
                go.Scatter(
                    x=self.results.index, y=self.results['Bench_Drawdown'] * 100,
                    mode='lines', name='Benchmark DD',
                    legend='legend2',
                    line=dict(color='orange', width=1),
                    hovertemplate="<b>Date:</b> %{x}<br><b>Bench DD:</b> %{y:.2f}%<extra></extra>"
                ), row=2, col=1
            )
        fig.update_yaxes(title_text="DD (%)", row=2, col=1)

        # --- Plot 3: Rolling Sharpe Ratio ---
        ann_factor = np.sqrt(self.periods_per_year)
        
        def get_rolling_sharpe(rets):
            roll_m = rets.rolling(rolling_n).mean()
            roll_s = rets.rolling(rolling_n).std()
            roll_sk = rets.rolling(rolling_n).skew()
            roll_ku = rets.rolling(rolling_n).kurt()
            
            sr_p = roll_m / roll_s
            sr_v = (1 / rolling_n) * (1 + 0.5 * sr_p**2 - roll_sk * sr_p + (roll_ku / 4) * sr_p**2)
            sr_se = np.sqrt(np.maximum(sr_v, 0))
            return sr_p * ann_factor, sr_se * ann_factor

        strat_sr, strat_se = get_rolling_sharpe(self.results['Net_Return'])
        
        x_ci = strat_sr.index.tolist() + strat_sr.index[::-1].tolist()
        y_ci = (strat_sr + 1.96 * strat_se).tolist() + (strat_sr - 1.96 * strat_se)[::-1].tolist()
        
        fig.add_trace(go.Scatter(
            x=x_ci, y=y_ci, fill='toself', fillcolor='rgba(0, 0, 255, 0.1)',
            line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='Strategy 95% PSR CI', 
            legend='legend3', showlegend=False
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=strat_sr.index, y=strat_sr, mode='lines', name='Strategy Sharpe',
            legend='legend3',
            line=dict(color='blue', width=2),
            hovertemplate="<b>Date:</b> %{x}<br><b>Strat Sharpe:</b> %{y:.2f}<extra></extra>"
        ), row=3, col=1)

        if self.benchmark_prices is not None:
            bench_sr, _ = get_rolling_sharpe(self.results['Bench_Return'])
            fig.add_trace(go.Scatter(
                x=bench_sr.index, y=bench_sr, mode='lines', name='Benchmark Sharpe',
                legend='legend3',
                line=dict(color='orange', width=1.5, dash='dot'),
                hovertemplate="<b>Date:</b> %{x}<br><b>Bench Sharpe:</b> %{y:.2f}<extra></extra>"
            ), row=3, col=1)
        
        fig.add_hline(y=0.0, line_width=1, line_dash="solid", line_color="black", opacity=0.3, row=3, col=1)
        fig.update_yaxes(title_text="Ann. Sharpe", row=3, col=1)

        pnl_bar_count = 0

        # --- Plot 4: PnL Attribution (Horizontal Bar Chart) ---
        if self.pnl_by_asset is not None and not self.pnl_by_asset.empty:
            # Keep only top 10 positive and bottom 10 negative contributors.
            top_10 = self.pnl_by_asset[self.pnl_by_asset > 0].nlargest(10)
            bottom_10 = self.pnl_by_asset[self.pnl_by_asset < 0].nsmallest(10).sort_values(ascending=False)

            # Desired visual order (top -> bottom):
            # largest positive -> ... -> 10th positive -> smallest negative -> ... -> largest negative.
            ordered_assets_top_to_bottom = list(top_10.index) + list(bottom_10.index)
            pnl_bar_count = len(ordered_assets_top_to_bottom)

            if pnl_bar_count > 0:
                y_pos_map = {
                    asset: pnl_bar_count - idx
                    for idx, asset in enumerate(ordered_assets_top_to_bottom)
                }
                tick_vals = list(range(1, pnl_bar_count + 1))
                tick_text = list(reversed(ordered_assets_top_to_bottom))

            if not top_10.empty:
                fig.add_trace(go.Bar(
                    x=top_10.values,
                    y=[y_pos_map[sym] for sym in top_10.index],
                    customdata=top_10.index,
                    orientation='h',
                    marker_color='rgba(40, 167, 69, 0.8)',
                    name='Top 10 PnL',
                    legend='legend4',
                    showlegend=True,
                    hovertemplate="<b>%{customdata}</b><br>Net PnL: $%{x:,.2f}<extra></extra>"
                ), row=4, col=1)

            if not bottom_10.empty:
                fig.add_trace(go.Bar(
                    x=bottom_10.values,
                    y=[y_pos_map[sym] for sym in bottom_10.index],
                    customdata=bottom_10.index,
                    orientation='h',
                    marker_color='rgba(220, 53, 69, 0.8)',
                    name='Bottom 10 PnL',
                    legend='legend4',
                    showlegend=True,
                    hovertemplate="<b>%{customdata}</b><br>Net PnL: $%{x:,.2f}<extra></extra>"
                ), row=4, col=1)
            
            fig.add_vline(x=0, line_width=1, line_color="black", opacity=0.5, row=4, col=1)
            fig.update_xaxes(title_text="Total Net PnL ($)", row=4, col=1)
            if pnl_bar_count > 0:
                fig.update_yaxes(
                    title_text="Asset",
                    tickmode='array',
                    tickvals=tick_vals,
                    ticktext=tick_text,
                    range=[0.5, pnl_bar_count + 0.5],
                    row=4,
                    col=1,
                )

        # --- Global Layout ---
        row1_legend_y = min(1.03, fig.layout.yaxis.domain[1] + 0.02)
        row2_legend_y = fig.layout.yaxis2.domain[1] + 0.02
        row3_legend_y = fig.layout.yaxis3.domain[1] + 0.02
        row4_legend_y = fig.layout.yaxis4.domain[1] + 0.02

        # Expand height when many assets are shown in row 4.
        fig_height = 1200 + max(0, pnl_bar_count - 12) * 20

        fig.update_layout(
            height=fig_height,
            hovermode="x unified",
            title_text="Backtest Observability Tearsheet",
            template="plotly_white",
            margin=dict(t=100, b=60, l=80, r=40),
            legend=dict(
                orientation="h", yanchor="bottom", y=row1_legend_y,
                xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"
            ),
            legend2=dict(
                orientation="h", yanchor="bottom", y=row2_legend_y,
                xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"
            ),
            legend3=dict(
                orientation="h", yanchor="bottom", y=row3_legend_y,
                xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"
            ),
            legend4=dict(
                orientation="h", yanchor="bottom", y=row4_legend_y,
                xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"
            )
        )

        fig.show()

    def get_holdings(self, timestamp: str) -> dict:
        """Retrieves the exact portfolio allocation for a given timestamp."""
        if self.executed_weights is None:
            raise ValueError("You must run() the backtest before checking holdings.")

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

        if ts not in index:
            idx = index.get_indexer([ts], method='pad')[0]
            if idx == -1:
                return {"Error": "Timestamp is before the backtest started."}
            ts = index[idx]
            print(f"Exact timestamp not found. Showing holdings for closest prior bar: {ts}")
            
        row = self.executed_weights.loc[ts]
        active_holdings = row[row > 0].to_dict()
        
        return active_holdings
    
    def get_holdings_history(self, output_format: str = 'long') -> pd.DataFrame:
        """Returns a complete history of all portfolio members over time."""
        if self.executed_weights is None:
            raise ValueError("You must run() the backtest before checking holdings.")
            
        if output_format == 'long':
            df = self.executed_weights.copy()
            df.index.name = 'Timestamp'
            
            long_df = df.reset_index().melt(
                id_vars='Timestamp', 
                var_name='Symbol', 
                value_name='Weight'
            )
            
            active_holdings = long_df[long_df['Weight'] > 0].copy()
            active_holdings.sort_values(by=['Timestamp', 'Weight'], ascending=[True, False], inplace=True)
            
            return active_holdings.set_index('Timestamp')
            
        elif output_format == 'wide':
            def get_active_members(row):
                active = row[row > 0]
                if active.empty:
                    return "CASH"
                return ", ".join([f"{sym}: {w*100:.1f}%" for sym, w in active.items()])
                
            members = self.executed_weights.apply(get_active_members, axis=1)
            return pd.DataFrame({'Portfolio_Members': members})
            
        else:
            raise ValueError("output_format must be 'long' or 'wide'.")