from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CostModel, PortfolioConfig
from .data import MarketData


@dataclass
class BacktestOutput:
    nav: pd.Series
    trades: pd.DataFrame
    rebalances: pd.DataFrame


def trade_commission(notional: float, costs: CostModel) -> float:
    if notional <= 0 or costs.commission_rate <= 0:
        return 0.0
    return max(costs.minimum_commission, notional * costs.commission_rate)


def _funding_scale(
    planned: list[tuple[str, float, float, float]],
    cash: float,
    costs: CostModel,
    minimum_order: float,
) -> float:
    def cash_needed(scale: float) -> float:
        needed = 0.0
        for _, notional, _, _ in planned:
            executed = notional * scale
            if executed < minimum_order:
                continue
            needed += (
                executed * (1.0 + costs.slippage_rate)
                + trade_commission(executed, costs)
            )
        return needed

    if not planned or cash <= 0:
        return 0.0
    if cash_needed(1.0) <= cash:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(50):
        middle = (low + high) / 2
        if cash_needed(middle) <= cash:
            low = middle
        else:
            high = middle
    return low


def simulate_target_weights(
    market: MarketData,
    targets: pd.DataFrame,
    costs: CostModel,
    config: PortfolioConfig,
    label: str,
) -> BacktestOutput:
    """Simulate long-only target weights with next-open execution."""
    required = {"execution_date", "code", "weight"}
    if not required.issubset(targets.columns):
        raise ValueError(f"Target file is missing columns: {required}")
    targets = targets.copy()
    targets["execution_date"] = pd.to_datetime(targets["execution_date"])
    targets["code"] = targets["code"].astype(str).str.zfill(6)
    if not targets.empty:
        sums = targets.groupby("execution_date")["weight"].sum()
        if not np.allclose(sums, 1.0, atol=1e-5):
            raise ValueError("Each target date must have weights summing to one")
    target_map = {
        date: frame.drop_duplicates("code").set_index("code")["weight"]
        for date, frame in targets.groupby("execution_date")
    }
    if not target_map:
        raise ValueError("No target dates supplied")

    calendar_adv20 = (
        market.amount.where(market.traded)
        .rolling(20, min_periods=15)
        .mean()
        .shift(1)
    )
    first_date = min(target_map)
    simulation_dates = market.calendar[market.calendar >= first_date]
    cash = config.initial_capital_cny
    holdings: dict[str, float] = {}
    nav_rows = []
    trade_rows = []
    rebalance_rows = []

    for date in simulation_dates:
        target_weights = target_map.get(date)
        if target_weights is not None:
            target_weights = target_weights[
                target_weights > 1e-10
            ].sort_values(ascending=False)
            target_codes = set(target_weights.index)
            open_row = market.open.loc[date]
            marked_row = market.close_marked.loc[date]
            prices = open_row.where(open_row.notna(), marked_row)
            pretrade_value = cash + sum(
                units * prices.get(code, np.nan)
                for code, units in holdings.items()
                if np.isfinite(prices.get(code, np.nan))
            )
            gross_traded = 0.0
            total_cost = 0.0
            blocked_orders = 0

            for code in sorted(set(holdings) | target_codes):
                units = holdings.get(code, 0.0)
                if units <= 0:
                    continue
                price = float(prices.get(code, np.nan))
                if not np.isfinite(price) or price <= 0:
                    continue
                desired_value = pretrade_value * float(
                    target_weights.get(code, 0.0)
                )
                reduction = max(0.0, units * price - desired_value)
                if reduction < config.minimum_order_notional_cny:
                    continue
                adv = float(calendar_adv20.at[date, code])
                capacity = (
                    config.adv_participation_cap * adv
                    if np.isfinite(adv)
                    else 0.0
                )
                planned_notional = min(reduction, capacity)
                if planned_notional < config.minimum_order_notional_cny:
                    blocked_orders += 1
                    trade_rows.append(
                        {
                            "date": date,
                            "code": code,
                            "side": "sell",
                            "desired_notional": reduction,
                            "executed_notional": 0.0,
                            "cost": 0.0,
                            "status": "capacity_limited",
                            "strategy": label,
                        }
                    )
                    continue
                if not bool(market.can_sell.at[date, code]):
                    blocked_orders += 1
                    reason = (
                        "limit_down"
                        if bool(market.limit_down.at[date, code])
                        else "not_trading"
                    )
                    trade_rows.append(
                        {
                            "date": date,
                            "code": code,
                            "side": "sell",
                            "desired_notional": reduction,
                            "executed_notional": 0.0,
                            "cost": 0.0,
                            "status": reason,
                            "strategy": label,
                        }
                    )
                    continue

                quantity = planned_notional / price
                commission = trade_commission(planned_notional, costs)
                stamp = planned_notional * costs.stamp_duty(date)
                slippage = planned_notional * costs.slippage_rate
                cash += planned_notional - commission - stamp - slippage
                holdings[code] = units - quantity
                if holdings[code] * price < config.minimum_order_notional_cny:
                    holdings.pop(code, None)
                gross_traded += planned_notional
                total_cost += commission + stamp + slippage
                trade_rows.append(
                    {
                        "date": date,
                        "code": code,
                        "side": "sell",
                        "desired_notional": reduction,
                        "executed_notional": planned_notional,
                        "cost": commission + stamp + slippage,
                        "status": (
                            "capacity_limited"
                            if capacity < reduction
                            else "filled"
                        ),
                        "strategy": label,
                    }
                )

            equity_after_sells = cash + sum(
                units * prices.get(code, np.nan)
                for code, units in holdings.items()
                if np.isfinite(prices.get(code, np.nan))
            )
            planned_buys: list[tuple[str, float, float, float]] = []
            for code, weight in target_weights.items():
                price = float(prices.get(code, np.nan))
                if not np.isfinite(price) or price <= 0:
                    blocked_orders += 1
                    continue
                units = holdings.get(code, 0.0)
                gap = max(0.0, equity_after_sells * float(weight) - units * price)
                adv = float(calendar_adv20.at[date, code])
                capacity = (
                    config.adv_participation_cap * adv
                    if np.isfinite(adv)
                    else 0.0
                )
                planned = min(gap, capacity)
                if planned < config.minimum_order_notional_cny:
                    if gap >= config.minimum_order_notional_cny:
                        blocked_orders += 1
                    continue
                if not bool(market.can_buy.at[date, code]):
                    blocked_orders += 1
                    reason = (
                        "limit_up"
                        if bool(market.limit_up.at[date, code])
                        else "not_trading"
                    )
                    trade_rows.append(
                        {
                            "date": date,
                            "code": code,
                            "side": "buy",
                            "desired_notional": gap,
                            "executed_notional": 0.0,
                            "cost": 0.0,
                            "status": reason,
                            "strategy": label,
                        }
                    )
                    continue
                planned_buys.append((code, planned, capacity, gap))

            scale = _funding_scale(
                planned_buys,
                cash=cash,
                costs=costs,
                minimum_order=config.minimum_order_notional_cny,
            )
            for code, planned, capacity, gap in planned_buys:
                notional = planned * scale
                if notional < config.minimum_order_notional_cny:
                    blocked_orders += 1
                    continue
                price = float(prices.at[code])
                commission = trade_commission(notional, costs)
                slippage = notional * costs.slippage_rate
                all_in = notional + commission + slippage
                if all_in > cash + 1.0:
                    raise RuntimeError(f"Buy exceeds cash on {date.date()}")
                cash -= all_in
                holdings[code] = holdings.get(code, 0.0) + notional / price
                gross_traded += notional
                total_cost += commission + slippage
                status = (
                    "filled"
                    if notional >= gap - 1e-6
                    else "capacity_or_cash_limited"
                )
                trade_rows.append(
                    {
                        "date": date,
                        "code": code,
                        "side": "buy",
                        "desired_notional": planned,
                        "executed_notional": notional,
                        "cost": commission + slippage,
                        "status": status,
                        "strategy": label,
                    }
                )

            rebalance_rows.append(
                {
                    "date": date,
                    "strategy": label,
                    "target_count": len(target_codes),
                    "held_count": len(holdings),
                    "gross_traded": gross_traded,
                    "one_way_turnover": (
                        gross_traded / (2.0 * pretrade_value)
                        if pretrade_value > 0
                        else np.nan
                    ),
                    "trading_cost": total_cost,
                    "blocked_orders": blocked_orders,
                    "cash_weight": (
                        cash
                        / (
                            cash
                            + sum(
                                units * marked_row.get(code, np.nan)
                                for code, units in holdings.items()
                                if np.isfinite(marked_row.get(code, np.nan))
                            )
                        )
                        if cash
                        + sum(
                            units * marked_row.get(code, np.nan)
                            for code, units in holdings.items()
                            if np.isfinite(marked_row.get(code, np.nan))
                        )
                        > 0
                        else np.nan
                    ),
                }
            )
            if cash < -1.0:
                raise RuntimeError(f"Negative cash after {date.date()}")
            if any(units < 0 for units in holdings.values()):
                raise RuntimeError(f"Negative holdings after {date.date()}")

        close_row = market.close_marked.loc[date]
        portfolio_value = cash + sum(
            units * close_row.get(code, np.nan)
            for code, units in holdings.items()
            if np.isfinite(close_row.get(code, np.nan))
        )
        if not np.isfinite(portfolio_value) or portfolio_value <= 0:
            raise RuntimeError(f"Invalid portfolio value on {date.date()}")
        nav_rows.append(
            {
                "date": date,
                "nav": portfolio_value / config.initial_capital_cny,
            }
        )

    nav = (
        pd.DataFrame(nav_rows)
        .set_index("date")["nav"]
        .sort_index()
        .rename(label)
    )
    return BacktestOutput(
        nav=nav,
        trades=pd.DataFrame(trade_rows),
        rebalances=pd.DataFrame(rebalance_rows),
    )
