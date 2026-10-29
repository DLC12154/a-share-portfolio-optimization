from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_research.risk import (  # noqa: E402
    estimate_shrunk_covariance,
    portfolio_volatility,
)


class RiskModelTests(unittest.TestCase):
    def test_shrunk_covariance_is_symmetric_psd_and_annualized(self) -> None:
        rng = np.random.default_rng(7)
        returns = pd.DataFrame(
            rng.normal(0.0, 0.01, size=(180, 4)),
            columns=["A", "B", "C", "D"],
        )
        covariance = estimate_shrunk_covariance(
            returns, returns.columns.tolist(), shrinkage=0.25
        )
        matrix = covariance.to_numpy()
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-10)
        self.assertTrue((np.diag(matrix) > 0).all())
        weights = pd.Series(0.25, index=covariance.index)
        self.assertGreater(portfolio_volatility(weights, covariance), 0.0)

    def test_incomplete_asset_is_removed(self) -> None:
        returns = pd.DataFrame(
            {
                "A": np.linspace(-0.01, 0.01, 100),
                "B": np.linspace(0.01, -0.01, 100),
                "C": np.nan,
            }
        )
        covariance = estimate_shrunk_covariance(
            returns, ["A", "B", "C"], shrinkage=0.20
        )
        self.assertEqual(covariance.index.tolist(), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
