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
    turnover_limited_target,
)
from .risk import (
    annualized_volatilities,
    estimate_eligible_universe_betas,
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
    previous_utility = pd.Series(dtype=float)
    previous_equal20_capped = pd.Series(dtype=float)
    previous_equal25_capped = pd.Series(dtype=float)

    grouped = list(signal.groupby("signal_date", sort=True))
    for index, (signal_date, date_frame) in enumerate(grouped, start=1):
        date_frame = date_frame.sort_values("alpha_rank")
        candidates = date_frame.loc[date_frame["selected"], "code"].tolist()
        scores = date_frame.set_index("code")["alpha_score"]
        execution_date = pd.Timestamp(date_frame["execution_date"].iloc[0])
        risk_returns = market.returns.loc[
            market.returns.index <= signal_date
        ].tail(config.risk_lookback_days)
        covariance = estimate_shrunk_covariance(
            risk_returns,
            candidates,
            shrinkage=config.covariance_shrinkage,
        )
        available_candidates = [
            code for code in candidates if code in covariance.index
        ]
        if len(available_candidates) < int(np.ceil(1.0 / config.max_weight)):
            raise ValueError(
                f"{signal_date.date()}: insufficient risk-model candidates"
            )
        candidates20 = available_candidates[:20]
        candidates25 = available_candidates[:25]
        equal20 = equal_weights(candidates20, config.max_weight)
        equal25 = equal_weights(candidates25, config.max_weight)
        strategies = {
            "equal_top20": equal20,
            "equal_top25": equal25,
            "equal_top20_turnover_cap": turnover_limited_target(
                equal20,
                previous_equal20_capped,
                max_one_way_turnover=config.max_one_way_turnover,
                max_weight=config.max_weight,
            ),
            "equal_top25_turnover_cap": turnover_limited_target(
                equal25,
                previous_equal25_capped,
                max_one_way_turnover=config.max_one_way_turnover,
                max_weight=config.max_weight,
            ),
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
        optimized_universe = sorted(
            set(available_candidates) | set(previous_optimized.index)
        )
        optimized_covariance = estimate_shrunk_covariance(
            risk_returns,
            optimized_universe,
            shrinkage=config.covariance_shrinkage,
        )
        eligible_market_codes = date_frame["code"].tolist()
        optimized_betas = estimate_eligible_universe_betas(
            risk_returns,
            optimized_covariance.index.tolist(),
            market_codes=eligible_market_codes,
        )
        optimization = optimize_weights(
            candidate_codes=available_candidates,
            alpha_score=scores,
            covariance=optimized_covariance,
            previous_weights=previous_optimized,
            max_weight=config.max_weight,
            annual_vol_target=config.annual_vol_target,
            max_one_way_turnover=config.max_one_way_turnover,
            betas=optimized_betas,
            beta_target=1.0,
            beta_tolerance=0.20,
            risk_aversion=0.0,
            turnover_penalty=0.0,
        )
        strategies["optimized"] = optimization.weights
        utility_universe = sorted(
            set(available_candidates) | set(previous_utility.index)
        )
        utility_covariance = estimate_shrunk_covariance(
            risk_returns,
            utility_universe,
            shrinkage=config.covariance_shrinkage,
        )
        utility_betas = estimate_eligible_universe_betas(
            risk_returns,
            utility_covariance.index.tolist(),
            market_codes=eligible_market_codes,
        )
        utility = optimize_weights(
            candidate_codes=available_candidates,
            alpha_score=scores,
            covariance=utility_covariance,
            previous_weights=previous_utility,
            max_weight=config.max_weight,
            annual_vol_target=config.annual_vol_target,
            max_one_way_turnover=config.max_one_way_turnover,
            betas=utility_betas,
            beta_target=1.0,
            beta_tolerance=0.20,
            risk_aversion=config.risk_aversion,
            turnover_penalty=config.turnover_penalty,
        )
        strategies["optimized_utility"] = utility.weights

        for strategy, weights in strategies.items():
            if strategy not in {"optimized", "optimized_utility"}:
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

        for strategy, result, risk_aversion, turnover_penalty, model_covariance, model_betas in [
            (
                "optimized",
                optimization,
                0.0,
                0.0,
                optimized_covariance,
                optimized_betas,
            ),
            (
                "optimized_utility",
                utility,
                config.risk_aversion,
                config.turnover_penalty,
                utility_covariance,
                utility_betas,
            ),
        ]:
            ex_ante_beta = (
                float(
                    model_betas.reindex(result.weights.index)
                    .fillna(1.0)
                    .dot(result.weights)
                )
                if len(result.weights)
                else np.nan
            )
            diagnostics.append(
                {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "strategy": strategy,
                    "candidate_count": len(candidates),
                    "risk_asset_count": len(model_covariance),
                    "optimizer_success": result.success,
                    "fallback": result.fallback,
                    "attempts": result.attempts,
                    "status": result.status,
                    "risk_aversion": risk_aversion,
                    "turnover_penalty": turnover_penalty,
                    "target_annual_volatility": config.annual_vol_target,
                    "effective_vol_target": result.effective_vol_target,
                    "realized_annual_volatility": result.annualized_volatility,
                    "target_one_way_turnover": result.target_one_way_turnover,
                    "max_one_way_turnover": config.max_one_way_turnover,
                    "ex_ante_universe_beta": ex_ante_beta,
                    "optimized_assets": int((result.weights > 1e-8).sum()),
                }
            )
        previous_optimized = optimization.weights
        previous_utility = utility.weights
        previous_equal20_capped = strategies["equal_top20_turnover_cap"]
        previous_equal25_capped = strategies["equal_top25_turnover_cap"]
        if index % 25 == 0 or index == len(grouped):
            print(
                f"Optimized {index}/{len(grouped)} signal dates; "
                f"alpha={optimization.success}/risk={utility.success}",
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
    success_rates = {
        str(strategy): float(rate)
        for strategy, rate in diagnostics_frame.groupby("strategy")[
            "optimizer_success"
        ]
        .mean()
        .items()
    }
    fallbacks = {
        str(strategy): int(count)
        for strategy, count in diagnostics_frame.groupby("strategy")[
            "fallback"
        ]
        .sum()
        .items()
    }
    return targets, diagnostics_frame, {
        "optimizer_success_rates": success_rates,
        "optimizer_fallbacks": fallbacks,
        "risk_aversion": config.risk_aversion,
        "turnover_penalty": config.turnover_penalty,
    }


def _drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def make_figures(
    navs: dict[str, pd.Series],
    rebalances: dict[str, pd.DataFrame],
    diagnostics: pd.DataFrame,
    performance: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    holdout = pd.concat(navs, axis=1).ffill()
    holdout = holdout.loc[
        (holdout.index >= HOLDOUT_START) & (holdout.index <= HOLDOUT_END)
    ]
    holdout = holdout.div(holdout.iloc[0])
    axis = holdout[
        [
            "optimized",
            "optimized_utility",
            "equal_top20_turnover_cap",
            "equal_top50",
            "benchmark",
        ]
    ].plot(figsize=(11, 5.5))
    axis.set_title("Time holdout portfolio value (2024-2025)")
    axis.set_xlabel("")
    axis.set_ylabel("Growth of 1.0")
    axis.figure.tight_layout()
    axis.figure.savefig(FIGURE_DIR / "holdout_nav.png", dpi=160)
    plt.close(axis.figure)

    drawdowns = holdout[
        [
            "optimized_utility",
            "equal_top20_turnover_cap",
            "benchmark",
        ]
    ].apply(_drawdown)
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
    axis = turnover[
        [
            "optimized_utility",
            "equal_top20_turnover_cap",
            "benchmark",
        ]
    ].plot(figsize=(11, 4.5), alpha=0.85)
    axis.set_title("One-way turnover at rebalance dates")
    axis.set_xlabel("")
    axis.set_ylabel("One-way turnover")
    axis.figure.tight_layout()
    axis.figure.savefig(FIGURE_DIR / "turnover.png", dpi=160)
    plt.close(axis.figure)

    alpha_only = diagnostics[diagnostics["strategy"] == "optimized"]
    utility = diagnostics[diagnostics["strategy"] == "optimized_utility"]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(
        utility["signal_date"],
        utility["target_annual_volatility"],
        label="Target volatility",
        linewidth=1.5,
    )
    axis.plot(
        alpha_only["signal_date"],
        alpha_only["realized_annual_volatility"],
        label="Alpha allocator volatility",
        linewidth=1.0,
        alpha=0.8,
    )
    axis.plot(
        utility["signal_date"],
        utility["realized_annual_volatility"],
        label="Risk-penalized utility volatility",
        linewidth=1.2,
    )
    axis.set_title("Optimizer ex-ante risk")
    axis.set_xlabel("")
    axis.set_ylabel("Annualized volatility")
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "optimizer_risk.png", dpi=160)
    plt.close(figure)

    yearly = performance[
        performance["period"].str.startswith("calendar_")
    ].copy()
    yearly["year"] = yearly["period"].str.removeprefix("calendar_")
    selected = yearly[
        yearly["portfolio"].isin(
            [
                "optimized_utility",
                "equal_top20_turnover_cap",
                "benchmark",
            ]
        )
    ].pivot(index="year", columns="portfolio", values="cagr")
    axis = selected.plot(kind="bar", figsize=(9, 4.5))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Calendar-year CAGR")
    axis.set_xlabel("")
    axis.set_ylabel("CAGR")
    axis.figure.tight_layout()
    axis.figure.savefig(FIGURE_DIR / "yearly_returns.png", dpi=160)
    plt.close(axis.figure)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        formatted = [
            str(value)
            if isinstance(value, str)
            else f"{float(value):.4f}"
            if pd.notna(value)
            else "n/a"
            for value in row
        ]
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def write_results_document(
    performance: pd.DataFrame,
    summary: dict,
    config: PortfolioConfig,
    signal_stats: pd.DataFrame,
) -> None:
    holdout = performance[performance["period"] == "holdout_2024_2025"].copy()
    event_excluded = performance[
        performance["period"] == "holdout_ex_2024_policy_rally"
    ].copy()
    yearly = performance[
        performance["period"].str.startswith("calendar_")
    ].copy()
    yearly["year"] = yearly["period"].str.removeprefix("calendar_")
    yearly_signal = signal_stats[
        signal_stats["stage"].str.startswith("calendar_")
    ].copy()
    yearly_signal["year"] = yearly_signal["stage"].str.removeprefix("calendar_")
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
    holdout_table = _markdown_table(holdout, columns)
    yearly_columns = [
        "year",
        "portfolio",
        "cagr",
        "sharpe",
        "max_drawdown",
        "mean_holdings",
        "mean_one_way_turnover",
    ]
    year_table = _markdown_table(yearly, yearly_columns)
    signal_table = _markdown_table(
        yearly_signal,
        ["year", "mean_rank_ic", "rank_icir", "positive_ic_ratio"],
    )
    event_columns = [
        "portfolio",
        "cagr",
        "sharpe",
        "max_drawdown",
    ]
    event_table = _markdown_table(event_excluded, event_columns)

    def metric(frame: pd.DataFrame, portfolio: str, column: str) -> float:
        value = frame.loc[
            frame["portfolio"] == portfolio, column
        ].iloc[0]
        return float(value)

    utility_2025 = yearly.loc[
        (yearly["year"] == "2025")
        & (yearly["portfolio"] == "optimized_utility")
    ].iloc[0]
    benchmark_2025 = yearly.loc[
        (yearly["year"] == "2025")
        & (yearly["portfolio"] == "benchmark")
    ].iloc[0]
    signal_2025 = yearly_signal.loc[yearly_signal["year"] == "2025"].iloc[0]
    markdown = [
        "# 组合优化研究结果",
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
        f"- 风险惩罚系数：{config.risk_aversion:.2f}",
        f"- 换手惩罚系数：{config.turnover_penalty:.2f}",
        "- Beta市场代理：当期全部合格沪深300历史成分股等权收益",
        "",
        "`optimized`只最大化Alpha并在硬约束内分配权重，"
        "`optimized_utility`额外把协方差风险和换手惩罚放入目标函数。"
        "后者是在已有结果审查后加入的透明扩展，不是预先注册的盲测策略。",
        "",
        "## 2024-2025整体结果",
        "",
        holdout_table,
        "",
        "## 分年结果",
        "",
        year_table,
        "",
        "## 分年Alpha有效性",
        "",
        signal_table,
        "",
        "## 排除2024年9-10月政策行情",
        "",
        "该敏感性分析将2024-09-10至2024-10-10及其后第一个交易日的收益统一归零，"
        "用于观察其余时间路径。它不是可交易规则。",
        "",
        event_table,
        "",
        "## 解释",
        "",
        f"2025年复合信号Mean RankIC为{signal_2025['mean_rank_ic']:.4f}，"
        f"同期风险惩罚组合CAGR为{utility_2025['cagr']:.2%}，"
        f"全市场等权净基准CAGR为{benchmark_2025['cagr']:.2%}。"
        "因此2024-2025的整体结果主要由2024年贡献，不能表述为跨年度稳定的Alpha。",
        "",
        f"时间外区间内，风险惩罚组合CAGR为"
        f"{metric(holdout, 'optimized_utility', 'cagr'):.2%}、"
        f"Sharpe为{metric(holdout, 'optimized_utility', 'sharpe'):.3f}、"
        f"最大回撤为{metric(holdout, 'optimized_utility', 'max_drawdown'):.2%}。"
        f"同期等权基准CAGR为{metric(holdout, 'benchmark', 'cagr'):.2%}、"
        f"Sharpe为{metric(holdout, 'benchmark', 'sharpe'):.3f}、"
        f"最大回撤为{metric(holdout, 'benchmark', 'max_drawdown'):.2%}；"
        "修正Beta后优化组合只有最大回撤更小，收益和Sharpe均落后。"
        f"同换手约束的Top 25等权组合CAGR为"
        f"{metric(holdout, 'equal_top25_turnover_cap', 'cagr'):.2%}、"
        f"Sharpe为{metric(holdout, 'equal_top25_turnover_cap', 'sharpe'):.3f}、"
        f"最大回撤为"
        f"{metric(holdout, 'equal_top25_turnover_cap', 'max_drawdown'):.2%}。"
        "风险惩罚降低了部分回撤，但没有改善收益，也没有稳定超过同规模简单等权组合，"
        "不能只与包含约300只股票的等权基准比较。",
        "",
        f"排除政策行情窗口后，风险惩罚组合CAGR为"
        f"{metric(event_excluded, 'optimized_utility', 'cagr'):.2%}、"
        f"Sharpe为{metric(event_excluded, 'optimized_utility', 'sharpe'):.3f}；"
        f"Top 25换手受限等权组合CAGR为"
        f"{metric(event_excluded, 'equal_top25_turnover_cap', 'cagr'):.2%}、"
        f"Sharpe为"
        f"{metric(event_excluded, 'equal_top25_turnover_cap', 'sharpe'):.3f}。",
        "",
        "## 优化器运行质量",
        "",
        *[
            f"- {strategy}成功求解比例：{rate:.1%}，回退次数："
            f"{summary['optimizer_fallbacks'].get(strategy, 0)}"
            for strategy, rate in summary["optimizer_success_rates"].items()
        ],
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
    strategy_targets["optimized_utility_gross"] = strategy_targets[
        "optimized_utility"
    ].copy()
    simulations: dict[str, BacktestOutput] = {}
    for label, frame in strategy_targets.items():
        print(f"Simulating {label}...", flush=True)
        chosen_costs = (
            config.zero_costs()
            if label.endswith("_gross")
            else costs
        )
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
        optimization_success_rates=optimizer_summary[
            "optimizer_success_rates"
        ],
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
            "The risk-penalized utility was added after the first holdout review and is not a blind strategy",
            "Beta uses the contemporaneous eligible-universe equal-weight proxy, not the official CSI 300 index",
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_figures(navs, rebalances, diagnostics, performance)
    write_results_document(
        performance,
        optimizer_summary,
        config,
        signal_stats,
    )
    print(performance.to_string(index=False), flush=True)
    print(
        "PASS: optimizer success rates "
        + ", ".join(
            f"{name}={rate:.1%}"
            for name, rate in optimizer_summary[
                "optimizer_success_rates"
            ].items()
        ),
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
