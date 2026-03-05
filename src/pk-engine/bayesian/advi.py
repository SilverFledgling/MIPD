"""
ADVI – Automatic Differentiation Variational Inference for PK parameters.

Approximates posterior p(eta|y) with a Gaussian q(eta; mu, sigma):
    min KL[q(eta) || p(eta|y)]

Equivalent to maximizing ELBO:
    ELBO = E_q[log p(y|eta)] - KL[q(eta) || p(eta)]

For Normal prior and Normal variational family, KL has closed form.
Likelihood evaluated via reparameterization trick.

Reference:
    - Kucukelbir et al. (2017), JMLR 18(1) – Automatic Differentiation VI
    - Margossian et al. (2022), CPT Pharmacometrics

Dependencies: numpy, scipy, pk.models, pk.solver, pk.population
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from pk.models import (
    DoseEvent, ErrorModel, ModelType, Observation, PKParams, PopPKModel,
)
from pk.solver import predict_concentrations
from pk.population import apply_iiv


# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────

@dataclass
class ADVIResult:
    """
    Result of ADVI posterior approximation.

    Attributes:
        mu:         Variational mean (eta space)
        sigma:      Variational std (eta space)
        params:     PK parameters at variational mean
        elbo:       Final ELBO value
        converged:  Whether optimizer converged
    """
    mu: NDArray[np.float64]
    sigma: NDArray[np.float64]
    params: PKParams
    elbo: float
    converged: bool


# ──────────────────────────────────────────────────────────────────
# ELBO computation
# ──────────────────────────────────────────────────────────────────

def _neg_elbo(
    phi: NDArray[np.float64],
    n_params: int,
    omega_diag: NDArray[np.float64],
    tv_params: PKParams,
    doses: list[DoseEvent],
    observations: list[Observation],
    error_model: ErrorModel,
    model_type: ModelType,
    n_mc: int = 10,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Negative ELBO for optimization.

    phi = [mu_0, ..., mu_d, log_sigma_0, ..., log_sigma_d]
    """
    if rng is None:
        rng = np.random.default_rng(0)

    mu = phi[:n_params]
    log_sigma = phi[n_params:]
    sigma = np.exp(log_sigma)

    obs_times = [obs.time for obs in observations]
    y_obs = np.array([obs.concentration for obs in observations])

    # KL divergence: KL[N(mu, sigma^2) || N(0, omega)]
    # = 0.5 * sum [sigma^2/omega + mu^2/omega - 1 - log(sigma^2/omega)]
    kl = 0.5 * np.sum(
        sigma ** 2 / omega_diag + mu ** 2 / omega_diag
        - 1.0 - np.log(sigma ** 2 / omega_diag + 1e-20)
    )

    # E_q[log p(y|eta)] via Monte Carlo with reparameterization
    log_lik = 0.0
    for _ in range(n_mc):
        eps = rng.standard_normal(n_params)
        eta = mu + sigma * eps  # Reparameterization

        try:
            ind_params = apply_iiv(tv_params, eta)
        except (ValueError, OverflowError):
            continue

        if ind_params.CL <= 0 or ind_params.V1 <= 0:
            log_lik -= 1e5
            continue

        try:
            c_pred = predict_concentrations(
                ind_params, doses, obs_times, model_type,
            )
        except (RuntimeError, ValueError):
            log_lik -= 1e5
            continue

        for j, obs in enumerate(observations):
            c_j = max(float(c_pred[j]), 1e-10)
            var_j = error_model.variance(c_j)
            var_j = max(var_j, 1e-10)
            residual = obs.concentration - c_j
            log_lik -= 0.5 * (residual ** 2 / var_j + np.log(var_j))

    log_lik /= n_mc
    elbo = log_lik - kl

    return -elbo  # Minimize negative ELBO


def run_advi(
    model: PopPKModel,
    tv_params: PKParams,
    doses: list[DoseEvent],
    observations: list[Observation],
    model_type: ModelType | None = None,
    n_mc: int = 10,
    max_iterations: int = 200,
    seed: int = 42,
) -> ADVIResult:
    """
    Run ADVI to approximate posterior.

    Args:
        model:           PopPK model
        tv_params:       Typical values
        doses:           Dose events
        observations:    TDM observations
        model_type:      PK model type
        n_mc:            MC samples for ELBO
        max_iterations:  Max optimization iterations
        seed:            Random seed

    Returns:
        ADVIResult with variational parameters
    """
    if not observations:
        raise ValueError("At least one observation required")
    if not doses:
        raise ValueError("At least one dose required")

    if model_type is None:
        model_type = model.model_type

    omega = np.array(model.omega_matrix, dtype=np.float64)
    omega_diag = np.diag(omega)
    n_params = omega.shape[0]
    rng = np.random.default_rng(seed)

    # Initial: mu=0, sigma=sqrt(omega)
    phi0 = np.concatenate([
        np.zeros(n_params),
        0.5 * np.log(omega_diag),  # log(sigma) = 0.5*log(omega)
    ])

    result = minimize(
        _neg_elbo,
        phi0,
        args=(
            n_params, omega_diag, tv_params, doses,
            observations, model.error_model, model_type, n_mc, rng,
        ),
        method="L-BFGS-B",
        options={"maxiter": max_iterations, "ftol": 1e-8},
    )

    mu = result.x[:n_params]
    sigma = np.exp(result.x[n_params:])
    params = apply_iiv(tv_params, mu)

    return ADVIResult(
        mu=mu,
        sigma=sigma,
        params=params,
        elbo=float(-result.fun),
        converged=bool(result.success),
    )
