from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .risk import annualized_volatilities, portfolio_volatility


@dataclass
class OptimizationResult:
    weights: pd.Series
    success: bool
    status: str
    attempts: int
    annualized_volatility: float
    target_one_way_turnover: float
    effective_vol_target: float
    fallback: bool


def capped_normalize(
    values: pd.Series,
    max_weight: float,
    required_total: float = 1.0,
) -> pd.Series:
    """Normalize positive values while iteratively enforcing a weight cap."""
    values = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    values = values.clip(lower=1e-12)
    if values.empty or len(values) * max_weight < required_total - 1e-10:
        raise ValueError("The candidate count cannot satisfy the weight cap")

    weights = pd.Series(0.0, index=values.index)
    remaining = list(values.index)
    remaining_total = required_total
    while remaining:
        current = values.loc[remaining]
        proposed = current / current.sum() * remaining_total
        over = proposed[proposed > max_weight + 1e-12]
        if over.empty:
            weights.loc[remaining] = proposed
            break
        fixed_codes = over.index.tolist()
        weights.loc[fixed_codes] = max_weight
        remaining_total -= max_weight * len(fixed_codes)
        remaining = [code for code in remaining if code not in set(fixed_codes)]
        if remaining_total <= 1e-12:
            break
    if not np.isclose(weights.sum(), required_total, atol=1e-8):
        raise RuntimeError("Capped normalization failed to allocate total weight")
    return weights


def equal_weights(codes: list[str], max_weight: float) -> pd.Series:
    return capped_normalize(pd.Series(1.0, index=codes), max_weight=max_weight)


def inverse_volatility_weights(
    covariance: pd.DataFrame,
    codes: list[str],
    max_weight: float,
) -> pd.Series:
    available = [code for code in codes if code in covariance.index]
    if len(available) < int(np.ceil(1.0 / max_weight)):
        raise ValueError("Not enough covariance assets for inverse-vol weights")
    volatility = annualized_volatilities(covariance).reindex(available)
    return capped_normalize(1.0 / volatility.clip(lower=1e-6), max_weight)


def score_weights(
    alpha_score: pd.Series,
    codes: list[str],
    max_weight: float,
) -> pd.Series:
    scores = alpha_score.reindex(codes).dropna()
    if scores.empty:
        raise ValueError("No scores available for score-weighted portfolio")
    shifted = scores - scores.min() + 0.05
    return capped_normalize(shifted, max_weight)


def turnover_limited_target(
    desired_weights: pd.Series,
    previous_weights: pd.Series,
    max_one_way_turnover: float,
    max_weight: float,
) -> pd.Series:
    """Move from the prior allocation toward a desired target without over-trading."""
    universe = sorted(set(desired_weights.index) | set(previous_weights.index))
    desired = desired_weights.reindex(universe, fill_value=0.0).astype(float)
    previous = previous_weights.reindex(universe, fill_value=0.0).astype(float)
    required = _one_way_turnover(desired.to_numpy(), previous.to_numpy())
    if required <= max_one_way_turnover:
        weights = desired
    else:
        scale = max_one_way_turnover / required
        weights = previous + scale * (desired - previous)
    weights = weights.clip(lower=0.0, upper=max_weight)
    weights = weights / weights.sum()
    return weights[weights.abs() > 1e-10]


def _one_way_turnover(
    weights: np.ndarray,
    previous: np.ndarray,
) -> float:
    return float(0.5 * np.abs(weights - previous).sum())


def optimize_weights(
    candidate_codes: list[str],
    alpha_score: pd.Series,
    covariance: pd.DataFrame,
    previous_weights: pd.Series,
    max_weight: float,
    annual_vol_target: float,
    max_one_way_turnover: float,
    betas: pd.Series | None = None,
    beta_target: float = 1.0,
    beta_tolerance: float = 0.20,
    risk_aversion: float = 0.0,
    turnover_penalty: float = 0.0,
) -> OptimizationResult:
    """Maximize alpha score subject to risk, weight and turnover constraints."""
    previous = previous_weights[previous_weights.abs() > 1e-10].astype(float)
    universe = sorted(set(candidate_codes) | set(previous.index))
    universe = [code for code in universe if code in covariance.index]
    candidates = [code for code in candidate_codes if code in universe]
    minimum_assets = int(np.ceil(1.0 / max_weight))
    if len(candidates) < minimum_assets:
        raise ValueError("Not enough candidates with covariance estimates")

    candidate_set = set(candidates)
    scores = alpha_score.reindex(candidates).astype(float)
    scores = scores.fillna(scores.min())
    full_scores = pd.Series(float(scores.min() - 1.0), index=universe)
    full_scores.loc[candidates] = scores
    previous_full = previous.reindex(universe, fill_value=0.0)
    covariance = covariance.reindex(index=universe, columns=universe)
    covariance_values = covariance.to_numpy()

    bounds = []
    lower_bounds = []
    upper_bounds = []
    for code in universe:
        if code in candidate_set:
            lower, upper = 0.0, max_weight
        else:
            lower, upper = (
                0.0,
                min(max_weight, float(previous_full.loc[code])),
            )
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        bounds.append((lower, upper))
    lower_array = np.asarray(lower_bounds, dtype=float)
    upper_array = np.asarray(upper_bounds, dtype=float)

    initial = pd.Series(0.0, index=universe)
    try:
        initial.loc[candidates] = equal_weights(candidates, max_weight)
    except ValueError:
        initial.loc[candidates] = 1.0 / len(candidates)
    if abs(initial.sum() - 1.0) > 1e-8:
        initial.loc[candidates] = 1.0 / len(candidates)
    initial_values = initial.clip(0.0, max_weight).to_numpy()

    beta_vector = None
    if betas is not None and len(betas):
        beta_vector = (
            betas.reindex(universe).fillna(1.0).clip(-3.0, 3.0).to_numpy()
        )

    def objective(weights: np.ndarray) -> float:
        regularization = 1e-7 * float(weights @ weights)
        risk_penalty = risk_aversion * float(
            weights @ covariance_values @ weights
        )
        turnover = _one_way_turnover(
            weights, previous_full.to_numpy()
        )
        return float(
            -(full_scores.to_numpy() @ weights)
            + risk_penalty
            + turnover_penalty * turnover
            + regularization
        )

    def risk_constraint(weights: np.ndarray, limit: float) -> float:
        variance = float(weights @ covariance_values @ weights)
        return limit * limit - variance

    def turnover_constraint(weights: np.ndarray) -> float:
        return max_one_way_turnover - _one_way_turnover(
            weights, previous_full.to_numpy()
        )

    def is_feasible(
        values: np.ndarray,
        vol_limit: float,
        *,
        tolerance: float = 2e-6,
    ) -> bool:
        if not np.isfinite(values).all():
            return False
        risk_value = float(values @ covariance_values @ values)
        beta_value = (
            float(beta_vector @ values)
            if beta_vector is not None
            else beta_target
        )
        return bool(
            abs(values.sum() - 1.0) <= tolerance * 10
            and np.all(values >= lower_array - tolerance)
            and np.all(values <= upper_array + tolerance)
            and risk_value <= vol_limit**2 + tolerance
            and _one_way_turnover(values, previous_full.to_numpy())
            <= max_one_way_turnover + tolerance
            and abs(beta_value - beta_target)
            <= beta_tolerance + tolerance * 10
        )

    def finalize_weights(values: np.ndarray) -> pd.Series:
        weights = pd.Series(values, index=universe).clip(
            lower=0.0, upper=upper_array
        )
        weights.loc[weights.abs() < 1e-12] = 0.0
        difference = 1.0 - weights.sum()
        if difference > 0:
            slack = pd.Series(upper_array, index=universe) - weights
            eligible = slack[slack > 1e-12]
            if eligible.empty:
                raise RuntimeError("No weight slack to complete portfolio")
            weights.loc[eligible.index] += (
                eligible / eligible.sum() * difference
            )
        elif difference < 0:
            reducible = weights - pd.Series(lower_array, index=universe)
            eligible = reducible[reducible > 1e-12]
            if eligible.empty:
                raise RuntimeError("No weight room to reduce portfolio")
            weights.loc[eligible.index] -= (
                eligible / eligible.sum() * (-difference)
            )
        return weights

    attempts = 0
    last_message = "not attempted"
    for multiplier in (1.0, 1.15, 1.35, 1.60):
        effective_target = annual_vol_target * multiplier
        constraints = [
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
            {
                "type": "ineq",
                "fun": lambda weights, limit=effective_target: risk_constraint(
                    weights, limit
                ),
            },
            {"type": "ineq", "fun": turnover_constraint},
        ]
        if (
            beta_vector is not None
            and np.nanstd(beta_vector) > 1e-8
        ):
            constraints.extend(
                [
                    {
                        "type": "ineq",
                        "fun": lambda weights: beta_tolerance
                        + (beta_vector @ weights - beta_target),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda weights: beta_tolerance
                        - (beta_vector @ weights - beta_target),
                    },
                ]
            )
        attempts += 1
        result = minimize(
            objective,
            initial_values,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9, "disp": False},
        )
        last_message = str(result.message)
        candidate = np.asarray(result.x, dtype=float)
        if is_feasible(candidate, effective_target):
            weights = finalize_weights(candidate)
            if not is_feasible(
                weights.to_numpy(),
                effective_target,
                tolerance=1e-5,
            ):
                last_message = (
                    "SLSQP result failed post-finalization constraint checks"
                )
                continue
            return OptimizationResult(
                weights=weights,
                success=True,
                status=last_message,
                attempts=attempts,
                annualized_volatility=portfolio_volatility(weights, covariance),
                target_one_way_turnover=_one_way_turnover(
                    weights.to_numpy(), previous_full.to_numpy()
                ),
                effective_vol_target=effective_target,
                fallback=False,
            )

    fallback_candidates: list[tuple[str, pd.Series]] = []
    try:
        fallback_candidates.append(
            (
                "inverse-vol",
                inverse_volatility_weights(
                    covariance, candidates, max_weight=max_weight
                ),
            )
        )
    except ValueError:
        pass
    try:
        fallback_candidates.append(
            ("equal-weight", equal_weights(candidates, max_weight=max_weight))
        )
    except ValueError:
        pass
    if previous.sum() > 0:
        fallback_candidates.append(("previous-weights", previous))

    for name, fallback in fallback_candidates:
        fallback_values = fallback.reindex(universe, fill_value=0.0).to_numpy()
        if is_feasible(
            fallback_values,
            annual_vol_target * 1.60,
            tolerance=1e-5,
        ):
            return OptimizationResult(
                weights=fallback,
                success=False,
                status=f"optimizer fallback to {name}: {last_message}",
                attempts=attempts,
                annualized_volatility=portfolio_volatility(
                    fallback, covariance
                ),
                target_one_way_turnover=_one_way_turnover(
                    fallback_values,
                    previous_full.to_numpy(),
                ),
                effective_vol_target=annual_vol_target * 1.60,
                fallback=True,
            )
    raise RuntimeError(
        "No optimizer or fallback portfolio satisfied all constraints: "
        f"{last_message}"
    )
