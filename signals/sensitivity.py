"""Generic parameter-sensitivity utilities for signal research.

This module is intentionally signal-agnostic:
- pass any signal class with a constructor and generate_weights(prices)
- pass 1 or 2 varying parameters in param_grid
- choose any strategy metric produced by VectorizedBacktester

Usage pattern:
1) results_df = run_parameter_sensitivity(...)
2) fig = plot_parameter_sensitivity(results_df)
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backtest import VectorizedBacktester


_METRIC_ALIASES = {
    "sharpe": "Sharpe Ratio",
    "sortino": "Sortino Ratio",
    "calmar": "Calmar Ratio",
    "psr": "Probabilistic Sharpe (%)",
    "return": "Ann. Return (%)",
    "ann_return": "Ann. Return (%)",
    "ann_vol": "Ann. Volatility (%)",
    "max_dd": "Max Drawdown (%)",
    "win_rate": "Win Rate (%)",
}


def _to_list(values: Iterable[Any]) -> List[Any]:
    """Convert a parameter grid iterable into a concrete list."""
    if isinstance(values, (str, bytes)):
        raise ValueError("Parameter grid values must be iterable collections, not strings")
    return list(values)


def _resolve_metric_name(requested_metric: str, available_metrics: Sequence[str]) -> str:
    """Resolve a user-provided metric key using aliases and exact matching."""
    if requested_metric in available_metrics:
        return requested_metric

    alias_key = requested_metric.strip().lower().replace(" ", "_")
    candidate = _METRIC_ALIASES.get(alias_key)
    if candidate and candidate in available_metrics:
        return candidate

    raise ValueError(
        "Unknown metric key '%s'. Available metrics: %s"
        % (requested_metric, ", ".join(available_metrics))
    )


def run_parameter_sensitivity(
    signal_class: Any,
    signal_base_params: Optional[Mapping[str, Any]],
    param_grid: Mapping[str, Iterable[Any]],
    prices: pd.DataFrame,
    benchmark_prices: Optional[pd.Series] = None,
    backtest_kwargs: Optional[Mapping[str, Any]] = None,
    metric: str = "Sharpe Ratio",
) -> pd.DataFrame:
    """Evaluate strategy sensitivity over a 1D or 2D parameter grid.

    Args:
        signal_class: Signal class type with constructor kwargs and generate_weights(prices).
        signal_base_params: Fixed kwargs always passed to signal constructor.
        param_grid: Dict of varying parameters. Supports exactly 1 or 2 keys in v1.
        prices: Price matrix passed to signal and backtester.
        benchmark_prices: Optional benchmark series for backtester.
        backtest_kwargs: Optional kwargs forwarded to VectorizedBacktester.
        metric: Strategy metric to optimize/plot. Supports aliases like sharpe/sortino/calmar.

    Returns:
        Tidy DataFrame with one row per parameter combination and key metrics.
    """
    if not isinstance(param_grid, Mapping) or len(param_grid) == 0:
        raise ValueError("param_grid must be a non-empty mapping of parameter -> iterable values")

    if len(param_grid) not in (1, 2):
        raise ValueError("v1 supports exactly 1 or 2 varying parameters")

    signal_base_params = dict(signal_base_params or {})
    backtest_kwargs = dict(backtest_kwargs or {})

    param_names = list(param_grid.keys())
    grid_values = [_to_list(param_grid[name]) for name in param_names]
    if any(len(vals) == 0 for vals in grid_values):
        raise ValueError("Each parameter in param_grid must contain at least one value")

    rows: List[Dict[str, Any]] = []
    metric_name_resolved: Optional[str] = None

    for combo in product(*grid_values):
        varying_params = dict(zip(param_names, combo))
        signal_params = {**signal_base_params, **varying_params}

        signal = signal_class(**signal_params)
        weights = signal.generate_weights(prices)

        backtester = VectorizedBacktester(
            prices=prices,
            weights=weights,
            benchmark_prices=benchmark_prices,
            **backtest_kwargs,
        )
        backtester.run()

        strategy_metrics = backtester.metrics.get("Strategy", {})
        if not strategy_metrics:
            raise ValueError("Backtest produced no Strategy metrics for parameter combo: %s" % varying_params)

        if metric_name_resolved is None:
            metric_name_resolved = _resolve_metric_name(metric, list(strategy_metrics.keys()))

        row: Dict[str, Any] = {
            "param_1_name": param_names[0],
            "param_1_value": combo[0],
            "param_2_name": param_names[1] if len(param_names) == 2 else None,
            "param_2_value": combo[1] if len(param_names) == 2 else np.nan,
            "metric_name": metric_name_resolved,
            "metric_value": strategy_metrics[metric_name_resolved],
            "Sharpe Ratio": strategy_metrics.get("Sharpe Ratio", np.nan),
            "Sortino Ratio": strategy_metrics.get("Sortino Ratio", np.nan),
            "Calmar Ratio": strategy_metrics.get("Calmar Ratio", np.nan),
            "Ann. Return (%)": strategy_metrics.get("Ann. Return (%)", np.nan),
            "Max Drawdown (%)": strategy_metrics.get("Max Drawdown (%)", np.nan),
            "Ann. Volatility (%)": strategy_metrics.get("Ann. Volatility (%)", np.nan),
            "Win Rate (%)": strategy_metrics.get("Win Rate (%)", np.nan),
        }
        rows.append(row)

    results_df = pd.DataFrame(rows)
    sort_cols = ["param_1_value"] if len(param_names) == 1 else ["param_1_value", "param_2_value"]
    results_df = results_df.sort_values(sort_cols).reset_index(drop=True)
    return results_df


def plot_parameter_sensitivity(
    results_df: pd.DataFrame,
    title: Optional[str] = None,
    show: bool = True,
) -> go.Figure:
    """Plot sensitivity as 2D line (1 parameter) or 3D surface (2 parameters)."""
    required_cols = {
        "param_1_name",
        "param_1_value",
        "param_2_name",
        "param_2_value",
        "metric_name",
        "metric_value",
    }
    missing = required_cols.difference(results_df.columns)
    if missing:
        raise ValueError("results_df missing required columns: %s" % sorted(missing))

    if results_df.empty:
        raise ValueError("results_df is empty")

    param_1_name = str(results_df["param_1_name"].iloc[0])
    metric_name = str(results_df["metric_name"].iloc[0])

    has_second_param = results_df["param_2_name"].notna().any() and results_df["param_2_value"].notna().any()

    if not has_second_param:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=results_df["param_1_value"],
                    y=results_df["metric_value"],
                    mode="lines+markers",
                    line=dict(width=2, color="#1f77b4"),
                    marker=dict(size=7),
                    hovertemplate=(
                        f"<b>{param_1_name}</b>: %{{x}}<br>"
                        f"<b>{metric_name}</b>: %{{y:.4f}}<extra></extra>"
                    ),
                )
            ]
        )
        fig.update_layout(
            title=title or f"Parameter Sensitivity: {param_1_name} vs {metric_name}",
            template="plotly_white",
            xaxis_title=param_1_name,
            yaxis_title=metric_name,
            hovermode="x unified",
        )
    else:
        param_2_name = str(results_df["param_2_name"].iloc[0])

        x_vals = np.sort(results_df["param_1_value"].unique())
        y_vals = np.sort(results_df["param_2_value"].unique())

        pivot = results_df.pivot_table(
            index="param_2_value",
            columns="param_1_value",
            values="metric_value",
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=y_vals, columns=x_vals)

        z_vals = pivot.values

        fig = go.Figure(
            data=[
                go.Surface(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    colorscale="Viridis",
                    colorbar=dict(title=metric_name),
                    hovertemplate=(
                        f"<b>{param_1_name}</b>: %{{x}}<br>"
                        f"<b>{param_2_name}</b>: %{{y}}<br>"
                        f"<b>{metric_name}</b>: %{{z:.4f}}<extra></extra>"
                    ),
                )
            ]
        )
        fig.update_layout(
            title=title or f"Parameter Sensitivity Surface: {metric_name}",
            template="plotly_white",
            scene=dict(
                xaxis_title=param_1_name,
                yaxis_title=param_2_name,
                zaxis_title=metric_name,
            ),
            margin=dict(l=0, r=0, t=60, b=0),
        )

    if show:
        fig.show()
    return fig
