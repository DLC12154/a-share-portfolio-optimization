from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    required_files = [
        PROCESSED / "signals.csv.gz",
        PROCESSED / "forward_returns.csv.gz",
        PROCESSED / "signal_quality.csv",
        PROCESSED / "target_weights.csv.gz",
        PROCESSED / "optimizer_diagnostics.csv",
        PROCESSED / "performance.csv",
        PROCESSED / "daily_nav.csv",
        PROCESSED / "trades.csv.gz",
        PROCESSED / "rebalances.csv",
        PROCESSED / "data_quality.json",
        PROCESSED / "run_metadata.json",
        ROOT / "figures" / "yearly_returns.png",
    ]
    for path in required_files:
        require(path.exists() and path.stat().st_size > 0, f"Missing output: {path}")

    manifest = json.loads(
        (RAW / "historical_dataset_manifest.json").read_text(encoding="utf-8")
    )
    require(
        sha256(RAW / "csi300_historical_membership.csv")
        == manifest["sha256"]["membership_file"],
        "Membership hash differs from the frozen manifest",
    )
    require(
        sha256(RAW / "csi300_prices_historical.parquet")
        == manifest["sha256"]["price_file"],
        "Price hash differs from the frozen manifest",
    )

    targets = pd.read_csv(
        PROCESSED / "target_weights.csv.gz",
        parse_dates=["signal_date", "execution_date"],
    )
    sums = targets.groupby(["execution_date", "strategy"])["weight"].sum()
    require(np.allclose(sums, 1.0, atol=1e-5), "Target weights do not sum to one")
    require(
        (targets["weight"] >= -1e-12).all(),
        "Negative target weight detected",
    )
    optimized = targets[
        targets["strategy"].isin(["optimized", "optimized_utility"])
    ]
    require(
        optimized["weight"].max() <= 0.05 + 1e-6,
        "Optimized portfolio exceeds the 5% stock cap",
    )
    for strategy in [
        "equal_top20_turnover_cap",
        "equal_top25_turnover_cap",
    ]:
        view = targets[targets["strategy"] == strategy]
        pivot = (
            view.pivot(index="execution_date", columns="code", values="weight")
            .fillna(0.0)
            .sort_index()
        )
        previous = pd.Series(0.0, index=pivot.columns)
        for _, current in pivot.iterrows():
            turnover = 0.5 * (current - previous).abs().sum()
            require(
                turnover <= 0.50 + 2e-5,
                f"{strategy} violates the 50% target-turnover cap",
            )
            previous = current

    diagnostics = pd.read_csv(PROCESSED / "optimizer_diagnostics.csv")
    require(
        bool(diagnostics["optimizer_success"].all()),
        "At least one optimizer date failed",
    )
    require(
        diagnostics["realized_annual_volatility"].max() <= 0.22 + 1e-5,
        "Ex-ante portfolio volatility exceeds the target",
    )
    require(
        diagnostics["target_one_way_turnover"].max() <= 0.50 + 1e-5,
        "Target one-way turnover exceeds the cap",
    )
    require(
        diagnostics["ex_ante_beta"].between(0.80 - 1e-5, 1.20 + 1e-5).all(),
        "Ex-ante portfolio beta breached its constraint",
    )

    navs = pd.read_csv(PROCESSED / "daily_nav.csv", parse_dates=["date"])
    require(
        (navs.drop(columns="date") > 0).all().all(),
        "Non-positive NAV detected",
    )
    trades = pd.read_csv(PROCESSED / "trades.csv.gz")
    require(
        (trades["executed_notional"] >= 0).all(),
        "Negative executed notional detected",
    )
    require((trades["cost"] >= 0).all(), "Negative trading cost detected")

    rebalances = pd.read_csv(PROCESSED / "rebalances.csv")
    require(
        (rebalances["cash_weight"] >= -1e-12).all(),
        "Negative cash weight detected",
    )

    performance = pd.read_csv(PROCESSED / "performance.csv")
    require(
        {
            "benchmark",
            "equal_top20_turnover_cap",
            "equal_top50",
            "optimized",
            "optimized_utility",
        }.issubset(
            set(performance["portfolio"])
        ),
        "Required comparison portfolios are missing",
    )
    require(
        {
            "calendar_2024",
            "calendar_2025",
            "holdout_ex_2024_policy_rally",
        }.issubset(set(performance["period"])),
        "Calendar-year or event-window diagnostics are missing",
    )
    signal_quality = pd.read_csv(PROCESSED / "signal_quality.csv")
    require(
        "calendar_2025" in set(signal_quality["stage"]),
        "2025 signal-quality diagnostic is missing",
    )

    print(
        "PASS: inputs, target weights, risk constraints, turnover, "
        "NAV, trades and performance artifacts validated"
    )
    print(
        f"dates={optimized['execution_date'].nunique()}, "
        f"target_rows={len(targets):,}, trades={len(trades):,}"
    )


if __name__ == "__main__":
    main()
