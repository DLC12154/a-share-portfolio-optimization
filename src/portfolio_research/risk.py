from __future__ import annotations

import numpy as np
import pandas as pd


def _make_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    floor = max(1e-12, float(np.max(np.abs(eigenvalues))) * 1e-10)
    values = np.clip(eigenvalues, floor, None)
    return (eigenvectors * values) @ eigenvectors.T


def estimate_shrunk_covariance(
    returns_window: pd.DataFrame,
    requested_codes: list[str],
    shrinkage: float,
    minimum_observations: int = 60,
) -> pd.DataFrame:
    """Estimate annualized covariance with a diagonal shrinkage target."""
    codes = [code for code in requested_codes if code in returns_window.columns]
    frame = returns_window[codes].copy()
    valid_counts = frame.notna().sum(axis=0)
    kept = valid_counts[valid_counts >= minimum_observations].index.tolist()
    if len(kept) < 2:
        raise ValueError("Fewer than two assets have enough return observations")
    frame = frame[kept].fillna(0.0)
    sample = np.cov(frame.to_numpy(dtype=float), rowvar=False, ddof=1)
    sample = np.atleast_2d(sample)
    diagonal = np.diag(np.diag(sample))
    shrunk = (1.0 - shrinkage) * sample + shrinkage * diagonal
    annualized = _make_positive_semidefinite(shrunk * 252.0)
    return pd.DataFrame(annualized, index=kept, columns=kept)


def portfolio_volatility(weights: pd.Series, covariance: pd.DataFrame) -> float:
    aligned = weights.reindex(covariance.index).fillna(0.0).to_numpy()
    variance = float(aligned @ covariance.to_numpy() @ aligned)
    return float(np.sqrt(max(variance, 0.0)))


def annualized_volatilities(covariance: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.sqrt(np.clip(np.diag(covariance.to_numpy()), 0.0, None)),
        index=covariance.index,
    )


def estimate_market_betas(
    returns_window: pd.DataFrame,
    codes: list[str],
) -> pd.Series:
    available = [code for code in codes if code in returns_window.columns]
    frame = returns_window[available].copy()
    market = frame.mean(axis=1)
    variance = market.var(ddof=1)
    if not np.isfinite(variance) or variance <= 0:
        return pd.Series(1.0, index=available)
    betas = {}
    for code in available:
        pair = pd.concat([frame[code], market.rename("market")], axis=1).dropna()
        if len(pair) < 30:
            betas[code] = 1.0
            continue
        covariance = pair[code].cov(pair["market"])
        betas[code] = float(covariance / variance)
    return pd.Series(betas)
