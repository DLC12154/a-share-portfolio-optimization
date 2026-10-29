from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "figures"
DOCS_DIR = ROOT / "docs"

PRIMARY_LOOKBACK_FACTORS = {
    "momentum_20": 20,
    "reversal_5": 5,
}

DEVELOPMENT_START = pd.Timestamp("2020-01-02")
DEVELOPMENT_END = pd.Timestamp("2023-12-29")
HOLDOUT_START = pd.Timestamp("2024-01-02")
HOLDOUT_END = pd.Timestamp("2025-12-31")


@dataclass(frozen=True)
class CostModel:
    commission_rate: float
    minimum_commission: float
    slippage_rate: float
    stamp_duty_pre_2023: float = 0.001
    stamp_duty_post_2023: float = 0.0005
    stamp_duty_changed: pd.Timestamp = pd.Timestamp("2023-08-28")

    def stamp_duty(self, date: pd.Timestamp) -> float:
        return (
            self.stamp_duty_post_2023
            if date >= self.stamp_duty_changed
            else self.stamp_duty_pre_2023
        )


@dataclass(frozen=True)
class PortfolioConfig:
    top_n: int = 50
    max_weight: float = 0.05
    annual_vol_target: float = 0.22
    max_one_way_turnover: float = 0.50
    risk_lookback_days: int = 120
    covariance_shrinkage: float = 0.20
    minimum_history_days: int = 120
    minimum_adv20_cny: float = 30_000_000.0
    adv_participation_cap: float = 0.05
    initial_capital_cny: float = 100_000_000.0
    minimum_order_notional_cny: float = 1_000.0
    commission_rate: float = 0.0003
    minimum_commission_cny: float = 5.0
    slippage_rate: float = 0.0005
    stamp_duty_pre_2023: float = 0.001
    stamp_duty_post_2023: float = 0.0005

    def realistic_costs(self) -> CostModel:
        return CostModel(
            commission_rate=self.commission_rate,
            minimum_commission=self.minimum_commission_cny,
            slippage_rate=self.slippage_rate,
            stamp_duty_pre_2023=self.stamp_duty_pre_2023,
            stamp_duty_post_2023=self.stamp_duty_post_2023,
        )

    def zero_costs(self) -> CostModel:
        return CostModel(
            commission_rate=0.0,
            minimum_commission=0.0,
            slippage_rate=0.0,
            stamp_duty_pre_2023=0.0,
            stamp_duty_post_2023=0.0,
        )

    def to_dict(self) -> dict:
        return asdict(self)
