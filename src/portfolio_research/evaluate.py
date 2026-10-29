from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    HOLDOUT_END,
    HOLDOUT_START,
)


def period_metrics(
    nav: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    before = nav.loc[nav.index < start]
    base = float(before.iloc[-1]) if not before.empty else 1.0
    segment = nav.loc[(nav.index >= start) & (nav.index <= end)]
    if segment.empty:
        return {}
    values = pd.concat(
        [pd.Series([base], index=[start - pd.Timedelta(days=1)]), segment]
    )
    returns = values.pct_change().dropna()
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    periods = max(len(returns), 1)
    cagr = float((1.0 + total_return) ** (252.0 / periods) - 1.0)
    volatility = (
        float(returns.std(ddof=1) * np.sqrt(252.0))
        if len(returns) > 1
        else np.nan
    )
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(252.0))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else np.nan
    )
    running_max = values.cummax()
    max_drawdown = float((values / running_max - 1.0).min())
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan
    return {
        "observations": int(len(segment)),
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "ending_nav": float(values.iloc[-1]),
    }


def _tracking_stats(
    nav: pd.Series,
    benchmark: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float, float]:
    frame = pd.concat([nav.rename("portfolio"), benchmark.rename("benchmark")], axis=1)
    frame = frame.loc[frame.index <= end].ffill()
    before = frame.loc[frame.index < start].tail(1)
    segment = frame.loc[(frame.index >= start) & (frame.index <= end)]
    if not before.empty:
        segment = pd.concat([before, segment])
    returns = segment.pct_change().dropna()
    active = returns["portfolio"] - returns["benchmark"]
    if len(active) < 2 or active.std(ddof=1) == 0:
        return np.nan, np.nan
    tracking_error = float(active.std(ddof=1) * np.sqrt(252.0))
    information_ratio = float(active.mean() / active.std(ddof=1) * np.sqrt(252.0))
    return tracking_error, information_ratio


def build_performance_table(
    navs: dict[str, pd.Series],
    rebalances: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    optimization_success_rate: float,
) -> pd.DataFrame:
    periods = {
        "development_2020_2023": (DEVELOPMENT_START, DEVELOPMENT_END),
        "holdout_2024_2025": (HOLDOUT_START, HOLDOUT_END),
        "full_2020_2025": (DEVELOPMENT_START, HOLDOUT_END),
    }
    benchmark = navs["benchmark"]
    rows = []
    for period, (start, end) in periods.items():
        for label, nav in navs.items():
            metrics = period_metrics(nav, start, end)
            adjustment = rebalances[label]
            in_period = adjustment[
                (adjustment["date"] >= start) & (adjustment["date"] <= end)
            ]
            target_view = targets[
                (targets["strategy"] == label)
                & (targets["execution_date"] >= start)
                & (targets["execution_date"] <= end)
            ]
            max_weights = (
                target_view.groupby("execution_date")["weight"].max()
                if not target_view.empty
                else pd.Series(dtype=float)
            )
            tracking_error, information_ratio = _tracking_stats(
                nav, benchmark, start, end
            )
            rows.append(
                {
                    "period": period,
                    "portfolio": label,
                    **metrics,
                    "tracking_error_vs_benchmark": tracking_error,
                    "information_ratio_vs_benchmark": information_ratio,
                    "mean_one_way_turnover": in_period["one_way_turnover"].mean(),
                    "mean_cash_weight": in_period["cash_weight"].mean(),
                    "mean_holdings": in_period["held_count"].mean(),
                    "mean_max_target_weight": max_weights.mean(),
                    "total_trading_cost_cny": in_period["trading_cost"].sum(),
                    "blocked_orders": in_period["blocked_orders"].sum(),
                    "optimizer_success_rate": optimization_success_rate
                    if label == "optimized"
                    else np.nan,
                }
            )
    performance = pd.DataFrame(rows)
    for period in periods:
        mask = performance["period"] == period
        benchmark_return = performance.loc[
            mask & (performance["portfolio"] == "benchmark"), "total_return"
        ].iloc[0]
        benchmark_sharpe = performance.loc[
            mask & (performance["portfolio"] == "benchmark"), "sharpe"
        ].iloc[0]
        performance.loc[mask, "excess_return_vs_benchmark"] = (
            performance.loc[mask, "total_return"] - benchmark_return
        )
        performance.loc[mask, "sharpe_difference_vs_benchmark"] = (
            performance.loc[mask, "sharpe"] - benchmark_sharpe
        )
    return performance
