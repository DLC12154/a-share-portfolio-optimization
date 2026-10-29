from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DEVELOPMENT_END,
    PortfolioConfig,
)
from .data import MarketData


FACTOR_COLUMNS = [
    "momentum_20",
    "reversal_5",
    "low_volatility_20",
    "low_turnover_20",
]


def _cross_sectional_rank(series: pd.Series, dates: pd.Series) -> pd.Series:
    ranked = series.groupby(dates).rank(method="average", pct=True)
    return ranked.sub(0.5)


def build_signal_panel(
    market: MarketData,
    membership: pd.DataFrame,
    config: PortfolioConfig,
    external_alpha_path: Path | None = None,
) -> pd.DataFrame:
    """Build a five-day signal panel using only information known at close t."""
    close = market.close_marked
    daily_returns = market.returns
    factors = {
        "momentum_20": close.div(close.shift(20)).sub(1),
        "reversal_5": close.div(close.shift(5)).sub(1).mul(-1),
        "low_volatility_20": daily_returns.rolling(
            20, min_periods=15
        ).std().mul(-1),
        "low_turnover_20": market.turnover.where(market.traded)
        .rolling(20, min_periods=15)
        .mean()
        .mul(-1),
    }
    adv20 = market.amount.where(market.traded).rolling(
        20, min_periods=15
    ).mean()
    active_history = market.traded.astype(float).rolling(
        config.minimum_history_days,
        min_periods=config.minimum_history_days,
    ).sum().shift(1)
    entry_date = pd.Series(market.calendar[1:], index=market.calendar[:-1])

    membership_map = (
        membership.groupby("query_date")["code"].agg(set).to_dict()
    )
    rows: list[dict] = []
    for raw_date, codes in membership_map.items():
        date = pd.Timestamp(raw_date)
        if date not in market.calendar:
            continue
        for code in sorted(codes):
            history = active_history.at[date, code]
            adv = adv20.at[date, code]
            if (
                not market.traded.at[date, code]
                or market.is_st.at[date, code]
                or not np.isfinite(history)
                or history < config.minimum_history_days
                or not np.isfinite(adv)
                or adv < config.minimum_adv20_cny
            ):
                continue
            row = {
                "signal_date": date,
                "execution_date": entry_date.get(date, pd.NaT),
                "code": code,
                "adv20_signal": adv,
                "active_history": history,
            }
            for name, matrix in factors.items():
                row[name] = matrix.at[date, code]
            rows.append(row)

    signal = pd.DataFrame(rows)
    if signal.empty:
        raise ValueError("No valid point-in-time signal rows were built")
    signal = signal.dropna(subset=["execution_date"]).copy()
    coverage = signal[FACTOR_COLUMNS].notna().sum(axis=1)
    signal = signal.loc[coverage >= 2].copy()

    for factor in FACTOR_COLUMNS:
        signal[f"{factor}_rank"] = _cross_sectional_rank(
            signal[factor], signal["signal_date"]
        )
    rank_columns = [f"{factor}_rank" for factor in FACTOR_COLUMNS]
    signal["factor_coverage"] = signal[rank_columns].notna().sum(axis=1)
    signal["composite_score"] = signal[rank_columns].mean(axis=1)

    if external_alpha_path is not None:
        external = pd.read_csv(
            external_alpha_path,
            dtype={"code": str},
            parse_dates=["date"],
        )
        required = {"date", "code", "score"}
        missing = required - set(external.columns)
        if missing:
            raise ValueError(
                f"External alpha file is missing columns: {sorted(missing)}"
            )
        external["code"] = external["code"].str.zfill(6)
        external = external.rename(
            columns={"date": "signal_date", "score": "alpha_score"}
        )
        if external.duplicated(["signal_date", "code"]).any():
            raise ValueError("External alpha has duplicate signal-date/code rows")
        signal = signal.merge(
            external[["signal_date", "code", "alpha_score"]],
            on=["signal_date", "code"],
            how="left",
            validate="one_to_one",
        )
        match_rate = signal["alpha_score"].notna().mean()
        if match_rate < 0.90:
            raise ValueError(
                f"External alpha code/date match rate is only {match_rate:.1%}"
            )
        signal["alpha_score"] = signal["alpha_score"].fillna(
            signal["alpha_score"].groupby(signal["signal_date"]).transform("median")
        )
        signal["alpha_source"] = "external"
    else:
        signal["alpha_score"] = signal["composite_score"]
        signal["alpha_source"] = "composite"

    signal["alpha_rank"] = (
        signal.groupby("signal_date")["alpha_score"]
        .rank(method="first", ascending=False)
    )
    signal["selected"] = signal["alpha_rank"] <= config.top_n
    signal["stage"] = np.where(
        signal["signal_date"] <= DEVELOPMENT_END,
        "development_2020_2023",
        "holdout_2024_2025",
    )
    signal = signal.sort_values(["signal_date", "alpha_rank"]).reset_index(drop=True)
    if signal["selected"].groupby(signal["signal_date"]).sum().min() < config.top_n:
        raise ValueError("At least one date has fewer than top_n candidates")
    return signal


def attach_forward_returns(
    signal: pd.DataFrame,
    market: MarketData,
) -> pd.DataFrame:
    """Attach future returns only after the point-in-time signal is frozen."""
    diagnostic = signal[["signal_date", "code"]].copy()
    forward = (
        market.open.shift(-6)
        .div(market.open.shift(-1))
        .sub(1)
    )
    diagnostic["forward_open_5"] = [
        forward.at[date, code]
        for date, code in diagnostic[["signal_date", "code"]].itertuples(
            index=False, name=None
        )
    ]
    return signal.merge(
        diagnostic,
        on=["signal_date", "code"],
        how="left",
        validate="one_to_one",
    )


def _signal_quality_rows(
    frame: pd.DataFrame,
    label: str,
) -> dict:
    valid = frame.dropna(subset=["alpha_score", "forward_open_5"])
    rank_ic = (
        valid.groupby("signal_date")
        .apply(
            lambda group: group["alpha_score"].corr(
                group["forward_open_5"], method="spearman"
            ),
            include_groups=False,
        )
        .dropna()
    )
    return {
        "stage": label,
        "signal_dates": int(frame["signal_date"].nunique()),
        "mean_rank_ic": float(rank_ic.mean()) if len(rank_ic) else np.nan,
        "rank_ic_std": float(rank_ic.std(ddof=1))
        if len(rank_ic) > 1
        else np.nan,
        "rank_icir": float(rank_ic.mean() / rank_ic.std(ddof=1))
        if len(rank_ic) > 1 and rank_ic.std(ddof=1) > 0
        else np.nan,
        "positive_ic_ratio": float(rank_ic.gt(0).mean())
        if len(rank_ic)
        else np.nan,
    }


def signal_quality(signal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, frame in signal.groupby("stage", sort=False):
        rows.append(_signal_quality_rows(frame, stage))
    years = signal["signal_date"].dt.year
    for year, frame in signal.groupby(years, sort=True):
        rows.append(_signal_quality_rows(frame, f"calendar_{year}"))
    return pd.DataFrame(rows)
