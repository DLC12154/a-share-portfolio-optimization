from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR


@dataclass
class MarketData:
    calendar: pd.DatetimeIndex
    codes: list[str]
    open: pd.DataFrame
    close: pd.DataFrame
    close_marked: pd.DataFrame
    preclose: pd.DataFrame
    amount: pd.DataFrame
    turnover: pd.DataFrame
    returns: pd.DataFrame
    traded: pd.DataFrame
    can_buy: pd.DataFrame
    can_sell: pd.DataFrame
    limit_up: pd.DataFrame
    limit_down: pd.DataFrame
    is_st: pd.DataFrame


def load_inputs(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_path = raw_dir / "csi300_prices_historical.parquet"
    membership_path = raw_dir / "csi300_historical_membership.csv"
    if not price_path.exists() or not membership_path.exists():
        raise FileNotFoundError(
            "Historical inputs are missing. Copy csi300_prices_historical.parquet "
            "and csi300_historical_membership.csv into data/raw."
        )
    prices = pd.read_parquet(price_path)
    membership = pd.read_csv(
        membership_path,
        dtype={"code": str},
        parse_dates=["query_date", "index_effective_date"],
    )
    prices["date"] = pd.to_datetime(prices["date"])
    prices["code"] = prices["code"].astype(str).str.zfill(6)
    membership["code"] = membership["code"].astype(str).str.zfill(6)
    return prices, membership


def _dense_field(panel: pd.DataFrame, field: str, index: pd.MultiIndex):
    return (
        panel.set_index(["date", "code"])[field]
        .reindex(index)
        .unstack("code")
    )


def _limit_band(codes: list[str], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame(0.10, index=calendar, columns=codes)
    chi_next_start = pd.Timestamp("2020-08-24")
    for code in codes:
        if code.startswith(("688", "689")):
            frame.loc[:, code] = 0.20
        elif code.startswith(("300", "301")):
            frame.loc[calendar < chi_next_start, code] = 0.10
            frame.loc[calendar >= chi_next_start, code] = 0.20
    return frame


def build_market(panel: pd.DataFrame) -> MarketData:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    calendar = pd.DatetimeIndex(sorted(panel["date"].unique()))
    codes = sorted(panel["code"].unique())
    index = pd.MultiIndex.from_product(
        [calendar, codes], names=["date", "code"]
    )
    open_price = _dense_field(panel, "open", index)
    close = _dense_field(panel, "close", index)
    preclose = _dense_field(panel, "preclose", index)
    volume = _dense_field(panel, "volume", index)
    amount = _dense_field(panel, "amount", index)
    turnover = _dense_field(panel, "turn", index)
    trade_status = _dense_field(panel, "tradestatus", index)
    st_flag = _dense_field(panel, "isST", index).ffill()

    traded = (
        (trade_status == 1)
        & volume.gt(0)
        & open_price.notna()
        & close.notna()
    )
    is_st = st_flag.eq(1) | st_flag.eq("1")
    limit = _limit_band(codes, calendar).mask(is_st, 0.05)
    open_change = open_price.div(preclose).sub(1)
    tolerance = 0.002
    limit_up = traded & open_change.ge(limit - tolerance)
    limit_down = traded & open_change.le(-limit + tolerance)
    close_marked = close.ffill()
    returns = close_marked.pct_change(fill_method=None)
    return MarketData(
        calendar=calendar,
        codes=codes,
        open=open_price,
        close=close,
        close_marked=close_marked,
        preclose=preclose,
        amount=amount,
        turnover=turnover,
        returns=returns,
        traded=traded,
        can_buy=traded & ~limit_up,
        can_sell=traded & ~limit_down,
        limit_up=limit_up,
        limit_down=limit_down,
        is_st=is_st,
    )


def data_quality_report(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    signal: pd.DataFrame,
) -> dict:
    snapshot_counts = membership.groupby("query_date")["code"].nunique()
    historical_codes = set(membership["code"])
    signal_dates = set(pd.to_datetime(signal["signal_date"]))
    membership_dates = set(pd.to_datetime(membership["query_date"]))
    duplicate_prices = int(prices.duplicated(["date", "code"]).sum())
    duplicate_signals = int(signal.duplicated(["signal_date", "code"]).sum())
    report = {
        "membership_snapshots": int(membership["query_date"].nunique()),
        "membership_rows": int(len(membership)),
        "all_snapshots_have_300_codes": bool((snapshot_counts == 300).all()),
        "historical_codes": int(len(historical_codes)),
        "price_rows": int(len(prices)),
        "price_codes": int(prices["code"].nunique()),
        "price_start": str(prices["date"].min().date()),
        "price_end": str(prices["date"].max().date()),
        "duplicate_price_keys": duplicate_prices,
        "duplicate_signal_keys": duplicate_signals,
        "signal_dates": int(len(signal_dates)),
        "membership_dates_without_signal": int(
            len(membership_dates - signal_dates)
        ),
        "signal_rows": int(len(signal)),
        "signal_rows_per_date": {
            "minimum": int(signal.groupby("signal_date").size().min()),
            "median": float(signal.groupby("signal_date").size().median()),
            "maximum": int(signal.groupby("signal_date").size().max()),
        },
        "null_factor_counts": {
            column: int(signal[column].isna().sum())
            for column in [
                "momentum_20",
                "reversal_5",
                "low_volatility_20",
                "low_turnover_20",
                "alpha_score",
            ]
        },
    }
    if duplicate_prices:
        raise AssertionError("Duplicate date-code rows in price panel")
    if duplicate_signals:
        raise AssertionError("Duplicate date-code rows in signal panel")
    if not report["all_snapshots_have_300_codes"]:
        raise AssertionError("At least one membership snapshot is incomplete")
    if report["membership_dates_without_signal"] > 6:
        raise AssertionError(
            "Too many historical membership dates lack a usable signal date"
        )
    return report
