from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import BacktestOutput, simulate_target_weights
from .config import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    DOCS_DIR,
    FIGURE_DIR,
    HOLDOUT_END,
    HOLDOUT_START,
    PROCESSED_DIR,
    PortfolioConfig,
)
from .data import (
    build_market,
    data_quality_report,
    load_inputs,
)
from .evaluate import build_performance_table, period_metrics
from .optimizer import (
    equal_weights,
    inverse_volatility_weights,
    optimize_weights,
    score_weights,
)
from .risk import (
    annualized_volatilities,
    estimate_market_betas,
    estimate_shrunk_covariance,
    portfolio_volatility,
)
from .signals import (
    attach_forward_returns,
    build_signal_panel,
    signal_quality,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_targets(
    signal: pd.DataFrame,
    market,
    config: PortfolioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    target_rows = []
    diagnostics = []
    previous_optimized = pd.Series(dtype=float)

    grouped = list(signal.groupby("signal_date", sort=True))
    for index, (signal_date, date_frame) in enumerate(grouped, start=1):
        date_frame = date_frame.sort_values("alpha_rank")
        candidates = date_frame.loc[date_frame["selected"], "code"].tolist()
        scores = date_frame.set_index("code")["alpha_score"]
        execution_date = pd.Timestamp(date_frame["execution_date"].iloc[0])
        risk_returns = market.returns.loc[
            market.returns.index <= signal_date
        ].tail(config.risk_lookback_days)
        risk_universe = sorted(
            set(candidates) | set(previous_optimized.index)
        )
        covariance = estimate_shrunk_covariance(
            risk_returns,
            risk_universe,
            shrinkage=config.covariance_shrinkage,
        )
        available_candidates = [
            code for code in candidates if code in covariance.index
        ]
        if len(available_candidates) < int(np.ceil(1.0 / config.max_weight)):
            raise ValueError(
                f"{signal_date.date()}: insufficient risk-model candidates"
            )
        betas = estimate_market_betas(
            risk_returns,
            covariance.index.tolist(),
        )

        strategies = {
            "equal_top50": equal_weights(
                available_candidates, config.max_weight
            ),
            "score_top50": score_weights(
                scores, available_candidates, config.max_weight
            ),
            "inverse_vol_top50": inverse_volatility_weights(
                covariance,
                available_candidates,
                config.max_weight,
            ),
        }
        optimization = optimize_weights(
            candidate_codes=available_candidates,
            alpha_score=scores,
            covariance=covariance,
            previous_weights=previous_optimized,
            max_weight=config.max_weight,
            annual_vol_target=config.annual_vol_target,
            max_one_way_turnover=config.max_one_way_turnover,
            betas=betas,
            beta_target=1.0,
            beta_tolerance=0.20,
        )
        strategies["optimized"] = optimization.weights

        for strategy, weights in strategies.items():
            weights = weights[weights > 1e-10]
            weights = weights / weights.sum()
            for code, weight in weights.items():
                target_rows.append(
                    {
                        "signal_date": signal_date,
                        "execution_date": execution_date,
                        "strategy": strategy,
                        "code": code,
                        "weight": float(weight),
                        "alpha_score": float(scores.get(code, np.nan)),
                    }
                )

        all_codes = date_frame["code"].tolist()
        benchmark_weights = equal_weights(all_codes, max_weight=1.0)
        for code, weight in benchmark_weights.items():
            target_rows.append(
                {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "strategy": "benchmark",
                    "code": code,
                    "weight": float(weight),
                    "alpha_score": float(scores.get(code, np.nan)),
                }
            )

        optimized_covariance = covariance.reindex(optimization.weights.index)
        ex_ante_beta = (
            float(
                betas.reindex(optimization.weights.index)
                .fillna(1.0)
                .dot(optimization.weights)
            )
            if len(optimization.weights)
            else np.nan
        )
        diagnostics.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "candidate_count": len(candidates),
                "risk_asset_count": len(covariance),
                "optimizer_success": optimization.success,
                "fallback": optimization.fallback,
                "attempts": optimization.attempts,
                "status": optimization.status,
                "target_annual_volatility": config.annual_vol_target,
                "effective_vol_target": optimization.effective_vol_target,
                "realized_annual_volatility": optimization.annualized_volatility,
                "target_one_way_turnover": optimization.target_one_way_turnover,
                "max_one_way_turnover": config.max_one_way_turnover,
                "ex_ante_beta": ex_ante_beta,
                "optimized_assets": int((optimization.weights > 1e-8).sum()),
            }
        )
        previous_optimized = optimization.weights
        if index % 25 == 0 or index == len(grouped):
            print(
                f"Optimized {index}/{len(grouped)} signal dates; "
                f"success={optimization.success}, fallback={optimization.fallback}",
                flush=True,
            )

    targets = pd.DataFrame(target_rows)
    targets["signal_date"] = pd.to_datetime(targets["signal_date"])
    targets["execution_date"] = pd.to_datetime(targets["execution_date"])
    diagnostics_frame = pd.DataFrame(diagnostics)
    if targets.empty or diagnostics_frame.empty:
        raise RuntimeError("No portfolio targets were produced")
    sums = targets.groupby(["execution_date", "strategy"])["weight"].sum()
    if not np.allclose(sums, 1.0, atol=1e-5):
        bad = sums[~np.isclose(sums, 1.0, atol=1e-5)]
        raise AssertionError(f"Invalid target weights: {bad.head().to_dict()}")
    return targets, diagnostics_frame, {
        "optimizer_success_rate": float(
            diagnostics_frame["optimizer_success"].mean()
        ),
        "optimizer_fallbacks": int(diagnostics_frame["fallback"].sum()),
    }


def _drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def make_figures(
    navs: dict[str, pd.Series],
    rebalances: dict[str, pd.DataFrame],
    diagnostics: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    holdout = pd.concat(navs, axis=1).ffill()
    holdout = holdout.loc[
        (holdout.index >= HOLDOUT_START) & (holdout.index <= HOLDOUT_END)
    ]
    holdout = holdout.div(holdout.iloc[0])
    axis = holdout[
        ["optimized", "equal_top50", "inverse_vol_top50", "benchmark"]
    ].plot(figsize=(11, 5.5))
    axis.set_title("Time holdout portfolio value (2024-2025)")
    axis.set_xlabel("")
    axis.set_ylabel("Growth of 1.0")
    axis.figure.tight_layout()
    axis.figure.savefig(FIGURE_DIR / "holdout_nav.png", dpi=160)
    plt.close(axis.figure)

    drawdowns = holdout[["optimized", "equal_top50", "benchmark"]].apply(
        _drawdown
    )
    axis = drawdowns.plot(figsize=(11, 4.5))
    axis.set_title("Time holdout drawdown")
    axis.set_xlabel("")
    axis.set_ylabel("Drawdown")
    axis.figure.tight_layout()
    axis.figure.savefig(FIGURE_DIR / "holdout_drawdown.png", dpi=160)
    plt.close(axis.figure)

    turnover = pd.DataFrame(
        {
            label: frame.set_index("date")["one_way_turnover"]
            for label, frame in rebalances.items()
        }
    ).sort_index()
    turnover = turnover.loc[
        (turnover.index >= HOLDOUT_START) & (turnover.index <= HOLDOUT_END)
    ]
    axis = turnover[["optimized", "equal_top50", "benchmark"]].plot(
        figsize=(11, 4.5), alpha=0.85
    )
    axis.set_title("One-way turnover at rebalance dates")
    axis.set_xlabel("")
    axis.set_ylabel("One-way turnover")
    axis.figure.tight_layout()
    axis.figure.savefig(FIGURE_DIR / "turnover.png", dpi=160)
    plt.close(axis.figure)

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(
        diagnostics["signal_date"],
        diagnostics["target_annual_volatility"],
        label="Target volatility",
        linewidth=1.5,
    )
    axis.plot(
        diagnostics["signal_date"],
        diagnostics["realized_annual_volatility"],
        label="Ex-ante portfolio volatility",
        linewidth=1.2,
    )
    axis.set_title("Optimizer ex-ante risk")
    axis.set_xlabel("")
    axis.set_ylabel("Annualized volatility")
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "optimizer_risk.png", dpi=160)
    plt.close(figure)


def write_results_document(
    performance: pd.DataFrame,
    summary: dict,
    config: PortfolioConfig,
) -> None:
    holdout = performance[
        performance["period"] == "holdout_2024_2025"
    ].copy()
    columns = [
        "portfolio",
        "cagr",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "excess_return_vs_benchmark",
        "mean_one_way_turnover",
        "mean_max_target_weight",
    ]
    table_lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in holdout[columns].itertuples(index=False, name=None):
        formatted = [
            str(value)
            if isinstance(value, str)
            else f"{float(value):.4f}"
            if pd.notna(value)
            else "n/a"
            for value in row
        ]
        table_lines.append("| " + " | ".join(formatted) + " |")
    markdown = [
        "# 时间外验证结果",
        "",
        "本次运行使用固定的四因子等权复合信号作为可复现Alpha基线。"
        "项目支持通过外部文件接入第二项目的滚动样本外预测分数。",
        "",
        "## 组合设置",
        "",
        f"- 候选股票数：{config.top_n}",
        f"- 单票最大权重：{config.max_weight:.1%}",
        f"- 年化波动率约束：{config.annual_vol_target:.1%}",
        f"- 单次最大单边换手：{config.max_one_way_turnover:.1%}",
        f"- 风险模型窗口：{config.risk_lookback_days}个交易日",
        f"- 协方差收缩系数：{config.covariance_shrinkage:.2f}",
        "",
        "## 2024-2025结果",
        "",
        *table_lines,
        "",
        "## 解释",
        "",
        "业绩结果只代表给定信号、约束和执行假设下的历史模拟。"
        "优化器的作用应结合Sharpe、最大回撤、换手和成本后收益共同判断，"
        "不能仅因为总收益变化就认定风险模型有效。",
        "",
        "## 优化器运行质量",
        "",
        f"- 成功求解比例：{summary['optimizer_success_rate']:.1%}",
        f"- 回退次数：{summary['optimizer_fallbacks']}",
        "",
    ]
    (DOCS_DIR / "results.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )


def run_pipeline(
    alpha_source: str = "composite",
    external_alpha: Path | None = None,
    output_dir: Path = PROCESSED_DIR,
    figure_dir: Path = FIGURE_DIR,
    config: PortfolioConfig | None = None,
) -> dict[str, pd.DataFrame | dict | dict[str, pd.Series]]:
    global FIGURE_DIR
    config = config or PortfolioConfig()
    if alpha_source == "external" and external_alpha is None:
        raise ValueError("--external-alpha is required for external alpha source")
    if alpha_source == "composite":
        external_alpha = None

    FIGURE_DIR = figure_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prices, membership = load_inputs()
    market = build_market(prices)
    print("Building point-in-time signals...", flush=True)
    signal = build_signal_panel(
        market,
        membership,
        config=config,
        external_alpha_path=external_alpha,
    )
    quality = data_quality_report(prices, membership, signal)
    diagnostic_signal = attach_forward_returns(signal, market)
    signal_stats = signal_quality(diagnostic_signal)
    print("Running risk-constrained portfolio construction...", flush=True)
    targets, diagnostics, optimizer_summary = build_targets(
        signal, market, config
    )

    costs = config.realistic_costs()
    strategy_targets = {
        label: frame.drop(columns=["strategy"])
        for label, frame in targets.groupby("strategy")
    }
    strategy_targets["optimized_gross"] = strategy_targets["optimized"].copy()
    simulations: dict[str, BacktestOutput] = {}
    for label, frame in strategy_targets.items():
        print(f"Simulating {label}...", flush=True)
        chosen_costs = config.zero_costs() if label == "optimized_gross" else costs
        simulations[label] = simulate_target_weights(
            market,
            frame,
            costs=chosen_costs,
            config=config,
            label=label,
        )
    navs = {label: output.nav for label, output in simulations.items()}
    trades = {label: output.trades for label, output in simulations.items()}
    rebalances = {
        label: output.rebalances for label, output in simulations.items()
    }
    performance = build_performance_table(
        navs,
        rebalances,
        targets,
        optimization_success_rate=optimizer_summary["optimizer_success_rate"],
    )

    signal.to_csv(
        output_dir / "signals.csv.gz", index=False, compression="gzip"
    )
    diagnostic_signal[["signal_date", "code", "forward_open_5"]].to_csv(
        output_dir / "forward_returns.csv.gz",
        index=False,
        compression="gzip",
    )
    signal_stats.to_csv(output_dir / "signal_quality.csv", index=False)
    targets.to_csv(
        output_dir / "target_weights.csv.gz",
        index=False,
        compression="gzip",
    )
    diagnostics.to_csv(
        output_dir / "optimizer_diagnostics.csv", index=False
    )
    performance.to_csv(output_dir / "performance.csv", index=False)
    pd.DataFrame(navs).to_csv(output_dir / "daily_nav.csv")
    pd.concat(trades.values(), ignore_index=True).to_csv(
        output_dir / "trades.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.concat(rebalances.values(), ignore_index=True).to_csv(
        output_dir / "rebalances.csv", index=False
    )
    (output_dir / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "alpha_source": alpha_source,
        "external_alpha": str(external_alpha) if external_alpha else None,
        "development_period": [
            str(DEVELOPMENT_START.date()),
            str(DEVELOPMENT_END.date()),
        ],
        "time_holdout_period": [
            str(HOLDOUT_START.date()),
            str(HOLDOUT_END.date()),
        ],
        "config": config.to_dict(),
        "optimizer": optimizer_summary,
        "inputs": {
            "prices_sha256": _sha256(
                Path(__file__).resolve().parents[2]
                / "data"
                / "raw"
                / "csi300_prices_historical.parquet"
            ),
            "membership_sha256": _sha256(
                Path(__file__).resolve().parents[2]
                / "data"
                / "raw"
                / "csi300_historical_membership.csv"
            ),
        },
        "known_limitations": [
            "No point-in-time industry classification in the verified run",
            "No 100-share lot rounding",
            "No permanent market impact or order-book queue model",
            "Baostock historical membership is provider-reconstructed",
            "2024-2025 is a retrospective time holdout, not a pre-registered OOS",
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_figures(navs, rebalances, diagnostics)
    write_results_document(performance, optimizer_summary, config)
    print(performance.to_string(index=False), flush=True)
    print(
        f"PASS: optimized successfully on "
        f"{optimizer_summary['optimizer_success_rate']:.1%} of dates",
        flush=True,
    )
    return {
        "performance": performance,
        "navs": navs,
        "targets": targets,
        "diagnostics": diagnostics,
        "quality": quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A-share multi-factor portfolio optimization research"
    )
    parser.add_argument(
        "--alpha-source",
        choices=["composite", "external"],
        default="composite",
    )
    parser.add_argument("--external-alpha", type=Path, default=None)
    args = parser.parse_args()
    run_pipeline(
        alpha_source=args.alpha_source,
        external_alpha=args.external_alpha,
    )


if __name__ == "__main__":
    main()
