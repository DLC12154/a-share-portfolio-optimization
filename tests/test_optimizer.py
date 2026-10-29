from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_research.optimizer import (  # noqa: E402
    capped_normalize,
    inverse_volatility_weights,
    optimize_weights,
    turnover_limited_target,
)
from portfolio_research.risk import portfolio_volatility  # noqa: E402


class OptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.codes = ["A", "B", "C", "D", "E"]
        self.covariance = pd.DataFrame(
            np.diag([0.04, 0.09, 0.16, 0.25, 0.36]),
            index=self.codes,
            columns=self.codes,
        )

    def test_capped_normalize_respects_sum_and_cap(self) -> None:
        values = pd.Series([10.0, 5.0, 3.0, 1.0, 1.0], index=self.codes)
        weights = capped_normalize(values, max_weight=0.30)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertTrue((weights <= 0.30 + 1e-10).all())

    def test_inverse_volatility_weights(self) -> None:
        weights = inverse_volatility_weights(
            self.covariance, self.codes, max_weight=0.35
        )
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertTrue((weights <= 0.35 + 1e-10).all())
        self.assertGreater(weights["A"], weights["E"])

    def test_optimizer_respects_constraints_and_turnover(self) -> None:
        scores = pd.Series([0.9, 0.6, 0.3, 0.1, -0.2], index=self.codes)
        previous = pd.Series(
            [0.20, 0.20, 0.20, 0.20, 0.20], index=self.codes
        )
        result = optimize_weights(
            candidate_codes=["A", "B", "C", "D"],
            alpha_score=scores,
            covariance=self.covariance,
            previous_weights=previous,
            max_weight=0.40,
            annual_vol_target=0.35,
            max_one_way_turnover=0.30,
            betas=pd.Series(1.0, index=self.codes),
            beta_tolerance=0.05,
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=6)
        self.assertTrue((result.weights <= 0.40 + 1e-7).all())
        self.assertLessEqual(result.target_one_way_turnover, 0.300001)
        self.assertLessEqual(
            portfolio_volatility(result.weights, self.covariance),
            result.effective_vol_target + 1e-5,
        )

    def test_turnover_limited_target_moves_gradually(self) -> None:
        target = pd.Series([0.5, 0.5, 0.0], index=["A", "B", "C"])
        previous = pd.Series([0.0, 0.5, 0.5], index=["A", "B", "C"])
        limited = turnover_limited_target(
            target,
            previous,
            max_one_way_turnover=0.10,
            max_weight=0.50,
        )
        turnover = 0.5 * (limited - previous).abs().sum()
        self.assertAlmostEqual(float(limited.sum()), 1.0, places=8)
        self.assertLessEqual(turnover, 0.100001)
        self.assertGreater(limited["C"], 0.0)

    def test_risk_penalty_changes_allocation(self) -> None:
        codes = ["A", "B"]
        covariance = pd.DataFrame(
            np.diag([0.50, 0.01]), index=codes, columns=codes
        )
        scores = pd.Series([1.0, 0.0], index=codes)
        previous = pd.Series(0.0, index=codes)
        common = dict(
            candidate_codes=codes,
            alpha_score=scores,
            covariance=covariance,
            previous_weights=previous,
            max_weight=1.0,
            annual_vol_target=1.0,
            max_one_way_turnover=1.0,
            betas=pd.Series([1.0, 1.0], index=codes),
            beta_tolerance=0.20,
        )
        alpha_only = optimize_weights(**common)
        risk_aware = optimize_weights(**common, risk_aversion=1.0)
        self.assertTrue(alpha_only.success)
        self.assertTrue(risk_aware.success)
        self.assertLess(
            portfolio_volatility(risk_aware.weights, covariance),
            portfolio_volatility(alpha_only.weights, covariance),
        )


if __name__ == "__main__":
    unittest.main()
