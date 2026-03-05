"""
BMA – Bayesian Model Averaging + Stacking for PK model selection.

Given K candidate models (e.g., different PopPK parameterizations):
    p(theta|y) = sum_k w_k * p(theta|y, M_k)

Weights can be computed via:
    1. BMA: w_k ∝ p(y|M_k) * p(M_k)       (marginal likelihood)
    2. Stacking: w_k = argmin KL(p || sum w_k * q_k)

Reference:
    - Hoeting et al. (1999), Statistical Science 14(4)
    - Yao et al. (2018), Bayesian Analysis 13(3) – Stacking
    - Uster et al. (2021), Clinical Pharmacology & Therapeutics

Dependencies: numpy, scipy, pk.models, pk.solver, pk.population
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize as scipy_minimize

from pk.models import (
    DoseEvent, ErrorModel, ModelType, Observation, PKParams, PopPKModel,
)
from pk.solver import predict_concentrations
from pk.population import apply_iiv


@dataclass
class BMAResult:
    """
    Result of Bayesian Model Averaging.

    Attributes:
        model_weights:    Weight for each model
        model_names:      Model names
        log_marginals:    Log marginal likelihood per model
        combined_params:  Weighted-average PK parameters
        best_model_idx:   Index of highest-weight model
        method:           'bma' or 'stacking'
    """
    model_weights: NDArray[np.float64]
    model_names: list[str]
    log_marginals: NDArray[np.float64]
    combined_params: PKParams
    best_model_idx: int
    method: str


def _laplace_log_marginal(
    model: PopPKModel,
    tv_params: PKParams,
    doses: list[DoseEvent],
    observations: list[Observation],
    model_type: ModelType,
) -> float:
    """
    Estimate log marginal likelihood via Laplace approximation.

    log p(y|M) ≈ log p(y|eta_MAP, M) + log p(eta_MAP|M) + d/2*log(2*pi) - 0.5*log|H|

    Simplified version: uses MAP + prior density.
    """
    omega = np.array(model.omega_matrix, dtype=np.float64)
    n_eta = omega.shape[0]
    omega_inv = np.linalg.inv(omega)

    # Find MAP via simple search (avoid circular import with MAP estimator)
    best_eta = np.zeros(n_eta)
    best_obj = float("inf")

    rng = np.random.default_rng(42)
    candidates = [np.zeros(n_eta)]
    for _ in range(20):
        candidates.append(rng.multivariate_normal(np.zeros(n_eta), omega))

    for eta in candidates:
        try:
            ind_params = apply_iiv(tv_params, eta)
        except (ValueError, OverflowError):
            continue

        if ind_params.CL <= 0 or ind_params.V1 <= 0:
            continue

        obs_times = [o.time for o in observations]
        try:
            c_pred = predict_concentrations(
                ind_params, doses, obs_times, model_type,
            )
        except (RuntimeError, ValueError):
            continue

        # Objective: prior + likelihood
        prior = 0.5 * float(eta @ omega_inv @ eta)
        lik = 0.0
        for j, obs in enumerate(observations):
            c_j = max(float(c_pred[j]), 1e-10)
            var_j = max(model.error_model.variance(c_j), 1e-10)
            residual = obs.concentration - c_j
            lik += 0.5 * (residual ** 2 / var_j + np.log(2 * np.pi * var_j))

        obj = prior + lik
        if obj < best_obj:
            best_obj = obj
            best_eta = eta.copy()

    # Laplace approximation: log p(y|M) ≈ -J(eta_MAP) + d/2 * log(2*pi)
    log_marginal = -best_obj + 0.5 * n_eta * np.log(2 * np.pi)
    # Subtract log|Omega| penalty (BIC-like regularization)
    log_marginal -= 0.5 * np.log(np.linalg.det(omega) + 1e-20)

    return float(log_marginal)


def run_bma(
    models: list[PopPKModel],
    tv_params_list: list[PKParams],
    doses: list[DoseEvent],
    observations: list[Observation],
    model_types: list[ModelType] | None = None,
    model_names: list[str] | None = None,
    prior_weights: NDArray[np.float64] | None = None,
    method: str = "bma",
) -> BMAResult:
    """
    Run Bayesian Model Averaging or Stacking.

    Args:
        models:          Candidate PopPK models
        tv_params_list:  Typical values for each model
        doses:           Dose events
        observations:    TDM observations
        model_types:     Model types (optional)
        model_names:     Model names (optional)
        prior_weights:   Prior model probabilities (uniform default)
        method:          'bma' or 'stacking'

    Returns:
        BMAResult with model weights and combined parameters
    """
    if not observations:
        raise ValueError("At least one observation required")
    if not doses:
        raise ValueError("At least one dose required")
    if len(models) < 2:
        raise ValueError("At least 2 models required for BMA")
    if len(models) != len(tv_params_list):
        raise ValueError("models and tv_params_list must have same length")

    K = len(models)

    if model_types is None:
        model_types = [m.model_type for m in models]
    if model_names is None:
        model_names = [m.name for m in models]
    if prior_weights is None:
        prior_weights = np.ones(K) / K

    # Compute log marginal likelihoods
    log_marginals = np.array([
        _laplace_log_marginal(
            models[k], tv_params_list[k], doses, observations, model_types[k],
        )
        for k in range(K)
    ])

    if method == "bma":
        # BMA weights: w_k ∝ p(y|M_k) * p(M_k)
        log_w = log_marginals + np.log(prior_weights + 1e-20)
        max_lw = np.max(log_w)
        weights = np.exp(log_w - max_lw)
        weights /= np.sum(weights)
    elif method == "stacking":
        # Stacking: optimize weights to minimize leave-one-out KL
        # Simplified: use softmax of log marginals
        log_w = log_marginals
        max_lw = np.max(log_w)
        weights = np.exp(log_w - max_lw)
        weights /= np.sum(weights)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bma' or 'stacking'")

    # Combined parameters (weighted average)
    combined_cl = sum(
        weights[k] * tv_params_list[k].CL for k in range(K)
    )
    combined_v1 = sum(
        weights[k] * tv_params_list[k].V1 for k in range(K)
    )
    combined_q = sum(
        weights[k] * tv_params_list[k].Q for k in range(K)
    )
    combined_v2 = sum(
        weights[k] * tv_params_list[k].V2 for k in range(K)
    )

    combined_params = PKParams(
        CL=float(combined_cl),
        V1=float(combined_v1),
        Q=float(combined_q),
        V2=float(combined_v2),
    )

    best_idx = int(np.argmax(weights))

    return BMAResult(
        model_weights=weights,
        model_names=model_names,
        log_marginals=log_marginals,
        combined_params=combined_params,
        best_model_idx=best_idx,
        method=method,
    )
