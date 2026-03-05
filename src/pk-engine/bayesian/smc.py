"""
SMC – Sequential Monte Carlo (Particle Filter) for PK parameters.

Algorithm:
    1. Initialize N particles from prior: eta_i ~ N(0, Omega)
    2. For each observation j:
        a. Weight particles by likelihood p(y_j | eta_i)
        b. Resample (systematic) if ESS < N/2
        c. Optional: Metropolis-Hastings rejuvenation
    3. Posterior = weighted particles

Advantages for MIPD:
    - Sequential: naturally handles arriving TDM data one-by-one
    - No gradient required (works with any likelihood)
    - Multimodal posterior support

Reference:
    - Del Moral, Doucet & Jasra (2006), JRSS-B 68(3)
    - Chopin (2002), Biometrika 89(3)

Dependencies: numpy, pk.models, pk.solver, pk.population
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pk.models import (
    DoseEvent, ErrorModel, ModelType, Observation, PKParams, PopPKModel,
)
from pk.solver import predict_concentrations
from pk.population import apply_iiv


@dataclass
class SMCResult:
    """
    Result of SMC posterior estimation.

    Attributes:
        particles:    Final particles (n_particles, n_params)
        weights:      Final normalized weights (n_particles,)
        ess_history:  ESS at each observation step
        params:       Weighted mean PK parameters
        n_resamples:  Number of resampling steps performed
    """
    particles: NDArray[np.float64]
    weights: NDArray[np.float64]
    ess_history: list[float]
    params: PKParams
    n_resamples: int


def _log_likelihood_single(
    eta: NDArray[np.float64],
    tv_params: PKParams,
    doses: list[DoseEvent],
    obs: Observation,
    error_model: ErrorModel,
    model_type: ModelType,
) -> float:
    """Log-likelihood for one observation at given eta."""
    try:
        ind_params = apply_iiv(tv_params, eta)
    except (ValueError, OverflowError):
        return -1e10

    if ind_params.CL <= 0 or ind_params.V1 <= 0:
        return -1e10

    try:
        c_pred = predict_concentrations(
            ind_params, doses, [obs.time], model_type,
        )
    except (RuntimeError, ValueError):
        return -1e10

    c_j = max(float(c_pred[0]), 1e-10)
    var_j = max(error_model.variance(c_j), 1e-10)
    residual = obs.concentration - c_j

    return -0.5 * (residual ** 2 / var_j + np.log(2 * np.pi * var_j))


def _systematic_resample(
    weights: NDArray[np.float64], rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Systematic resampling."""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n

    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # Ensure no floating point issues

    indices = np.searchsorted(cumsum, positions)
    return indices


def run_smc(
    model: PopPKModel,
    tv_params: PKParams,
    doses: list[DoseEvent],
    observations: list[Observation],
    model_type: ModelType | None = None,
    n_particles: int = 500,
    ess_threshold: float = 0.5,
    seed: int = 42,
) -> SMCResult:
    """
    Run SMC particle filter for sequential Bayesian PK estimation.

    Args:
        model:          PopPK model
        tv_params:      Typical values
        doses:          Dose events
        observations:   TDM observations (processed sequentially)
        model_type:     PK model type
        n_particles:    Number of particles
        ess_threshold:  Resample when ESS < threshold * N
        seed:           Random seed

    Returns:
        SMCResult with weighted particles
    """
    if not observations:
        raise ValueError("At least one observation required")
    if not doses:
        raise ValueError("At least one dose required")

    if model_type is None:
        model_type = model.model_type

    rng = np.random.default_rng(seed)
    omega = np.array(model.omega_matrix, dtype=np.float64)
    n_eta = omega.shape[0]

    # Initialize particles from prior
    particles = rng.multivariate_normal(
        np.zeros(n_eta), omega, size=n_particles,
    )
    log_weights = np.zeros(n_particles)
    ess_history: list[float] = []
    n_resamples = 0

    # Process observations sequentially
    for obs in observations:
        # Weight update
        for i in range(n_particles):
            ll = _log_likelihood_single(
                particles[i], tv_params, doses, obs,
                model.error_model, model_type,
            )
            log_weights[i] += ll

        # Normalize weights
        max_lw = np.max(log_weights)
        weights = np.exp(log_weights - max_lw)
        weights /= np.sum(weights)

        # ESS
        ess = 1.0 / np.sum(weights ** 2)
        ess_history.append(float(ess))

        # Resample if ESS too low
        if ess < ess_threshold * n_particles:
            indices = _systematic_resample(weights, rng)
            particles = particles[indices].copy()
            log_weights = np.zeros(n_particles)

            # Jitter (random walk rejuvenation)
            jitter_scale = 0.1 * np.sqrt(np.diag(omega))
            particles += rng.normal(0, jitter_scale, particles.shape)
            n_resamples += 1

    # Final weights
    max_lw = np.max(log_weights)
    weights = np.exp(log_weights - max_lw)
    weights /= np.sum(weights)

    # Weighted mean
    mean_eta = np.average(particles, weights=weights, axis=0)
    params = apply_iiv(tv_params, mean_eta)

    return SMCResult(
        particles=particles,
        weights=weights,
        ess_history=ess_history,
        params=params,
        n_resamples=n_resamples,
    )
