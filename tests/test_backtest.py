from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_research.backtest import simulate_target_weights  # noqa: E402
from portfolio_research.config import PortfolioConfig  # noqa: E402
from portfolio_research.data import build_market  # noqa: E402


class BacktestTests(unittest.TestCase):
    def _market(self):
        dates = pd.bdate_range("2024-01-02", periods=40)
        rows = []
        for date_index, date in enumerate(dates):
            for code, base in [("000001", 10.0), ("000002", 20.0)]:
                close = base * (1.0 + 0.001 * date_index)
                rows.append(
                    {
                        "date": date,
                        "code": code,
                        "open": close,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "close": close,
                        "preclose": base * (1.0 + 0.001 * (date_index - 1)),
                        "volume": 1_000_000.0,
                        "amount": 100_000_000.0,
                        "turn": 1.0,
                        "tradestatus": 1.0,
                        "isST": 0.0,
                    }
                )
        return build_market(pd.DataFrame(rows))

    def test_target_weights_trade_next_open_and_charge_costs(self) -> None:
        market = self._market()
        execution_date = market.calendar[20]
        targets = pd.DataFrame(
            {
                "execution_date": [execution_date, execution_date],
                "code": ["000001", "000002"],
                "weight": [0.5, 0.5],
            }
        )
        config = PortfolioConfig(initial_capital_cny=20_000.0)
        output = simulate_target_weights(
            market,
            targets,
            costs=config.realistic_costs(),
            config=config,
            label="test",
        )
        self.assertEqual(output.nav.index.min(), execution_date)
        self.assertEqual(len(output.trades), 2)
        self.assertTrue((output.trades["executed_notional"] > 0).all())
        self.assertGreater(output.trades["cost"].sum(), 0.0)
        self.assertTrue(
            output.trades["status"].eq("capacity_or_cash_limited").all()
        )
        executed = output.trades["executed_notional"].to_numpy()
        self.assertLess(abs(executed[0] - executed[1]), 100.0)
        self.assertLess(output.nav.iloc[0], 1.0)
        self.assertGreater(output.nav.iloc[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
