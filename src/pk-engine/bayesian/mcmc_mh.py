"""
MCMC Estimator — Pure-Python Metropolis-Hastings posterior sampling.

Fallback implementation when JAX/NumPyro is not available.

Algorithm (Random Walk Metropolis-Hastings):
    1. Start at η₀ = MAP estimate (warm start)
    2. For each iteration:
       a. Propose η* = η_current + ε,  ε ~ N(0, proposal_cov)
       b. Accept/reject: r = p(y|η*) × p(η*) / [p(y|η_current) × p(η_current)]
       c. Accept if r > uniform(0,1)
    3. Discard warmup, thin, compute summary statistics

Reference:
    - Metropolis et al. (1953), J Chem Phys
    - Robert & Casella (2004), Monte Carlo Statistical Methods
    - MCMCcode.R (Vancomycin nhi khoa) — uses equivalent MH algorithm

Dependencies: numpy, scipy (only for ODE solver), pk.models, pk.population
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pk.models import (
    DoseEvent, Observation, PKParams, ModelType, PopPKModel, ErrorModel,
)
from pk.analytical import predict_analytical
from pk.population import apply_iiv


@dataclass
class MCMCResultMH:
    """
    Result of Metropolis-Hastings MCMC posterior sampling.

    Attributes:
        posterior_eta:    Posterior samples (n_samples, n_params)
        posterior_params: Summary statistics per PK parameter
        map_params:       Best (MAP) parameter set from chain
        acceptance_rate:  Fraction of proposals accepted
        n_samples:        Number of post-warmup samples
        converged:        Whether chain appears converged
        rhat:             Gelman-Rubin R-hat (if 2+ chains)
        ess:              Effective sample size
        n_divergences:    Number of rejected proposals (for diagnostics)
    """
    posterior_eta: NDArray[np.float64]
    posterior_params: dict[str, dict]
    map_params: PKParams
    acceptance_rate: float
    n_samples: int
    converged: bool
    rhat: float
    ess: float
    n_divergences: int


def _log_posterior(
    eta: NDArray[np.float64],
    tv_params: PKParams,
    omega_inv: NDArray[np.float64],
    doses: list[DoseEvent],
    observations: list[Observation],
    error_model: ErrorModel,
    model_type: ModelType,
) -> float:
    """
    Compute log-posterior(η) = log-likelihood + log-prior.

    log-prior: -0.5 × η^T × Ω^(-1) × η
    log-likelihood: Σ_j log N(y_j | C_pred(η), σ²_j)
    """
    # Prior: η ~ N(0, Ω)
    log_prior = -0.5 * eta @ omega_inv @ eta

    # Likelihood
    try:
        ind_params = apply_iiv(tv_params, eta)
    except (ValueError, OverflowError):
        return -1e10

    if ind_params.CL <= 0 or ind_params.V1 <= 0:
        return -1e10

    try:
        times = [obs.time for obs in observations]
        c_pred_arr = predict_analytical(
            ind_params, doses, times, model_type,
        )
    except Exception:
        return -1e10

    log_likelihood = 0.0
    for j, obs in enumerate(observations):
        c_j = max(float(c_pred_arr[j]), 1e-10)
        var_j = max(error_model.variance(c_j), 1e-10)
        residual = obs.concentration - c_j
        log_likelihood += -0.5 * (residual ** 2 / var_j + np.log(2 * np.pi * var_j))

    return log_prior + log_likelihood


def _compute_ess(samples: NDArray[np.float64]) -> float:
    """Estimate Effective Sample Size using autocorrelation."""
    n = len(samples)
    if n < 10:
        return float(n)

    mean = np.mean(samples)
    var = np.var(samples)
    if var < 1e-20:
        return float(n)

    # Compute autocorrelation
    max_lag = min(n // 2, 100)
    acf = np.zeros(max_lag)
    centered = samples - mean
    for lag in range(max_lag):
        acf[lag] = np.mean(centered[:n - lag] * centered[lag:]) / var

    # Sum autocorrelations until first negative pair
    tau = 1.0
    for lag in range(1, max_lag - 1, 2):
        rho_pair = acf[lag] + acf[lag + 1]
        if rho_pair < 0:
            break
        tau += 2 * rho_pair

    return max(1.0, float(n / tau))


def run_mcmc_mh(
    model: PopPKModel,
    tv_params: PKParams,
    doses: list[DoseEvent],
    observations: list[Observation],
    *,
    n_warmup: int = 500,
    n_samples: int = 1000,
    n_chains: int = 2,
    proposal_scale: float = 0.1,
    seed: int = 42,
) -> MCMCResultMH:
    """
    Run Metropolis-Hastings MCMC for PK parameter estimation.

    This is a pure-Python implementation that does not require JAX.
    Algorithm follows the same logic as MCMCcode.R (Vancomycin).

    Args:
        model:          PopPK model definition
        tv_params:      Covariate-adjusted typical values
        doses:          Dose events
        observations:   TDM observations
        n_warmup:       Warmup (burn-in) iterations per chain
        n_samples:      Post-warmup samples per chain
        n_chains:       Number of chains (for R-hat computation)
        proposal_scale: Scale factor for proposal distribution
        seed:           Random seed

    Returns:
        MCMCResultMH with posterior samples and summary statistics
    """
    if not observations:
        raise ValueError("At least one observation required for MCMC")

    omega = np.array(model.omega_matrix, dtype=np.float64)
    omega_inv = np.linalg.inv(omega)
    n_eta = omega.shape[0]
    model_type = model.model_type

    # Proposal covariance: scaled version of population omega
    proposal_cov = proposal_scale ** 2 * omega
    proposal_chol = np.linalg.cholesky(proposal_cov)

    rng = np.random.default_rng(seed)
    all_chains = []

    for chain_idx in range(n_chains):
        chain_rng = np.random.default_rng(seed + chain_idx * 1000)

        # Initialize from overdispersed prior
        eta_current = chain_rng.multivariate_normal(
            np.zeros(n_eta), omega * 0.5,
        )
        lp_current = _log_posterior(
            eta_current, tv_params, omega_inv,
            doses, observations, model.error_model, model_type,
        )

        chain_samples = []
        n_accepted = 0
        total_iters = n_warmup + n_samples

        # Adaptive proposal: start with initial scale, adapt during warmup
        # Target acceptance rate ~23% (Roberts et al., 1997, optimal for RW-MH)
        current_scale = proposal_scale
        adapt_interval = 50  # Adapt every 50 iterations
        warmup_accepts = 0
        warmup_iters = 0

        for it in range(total_iters):
            # Recompute proposal Cholesky with current scale
            if it > 0 and it < n_warmup and it % adapt_interval == 0 and warmup_iters > 0:
                current_rate = warmup_accepts / warmup_iters
                # Adapt scale: increase if acceptance too high, decrease if too low
                if current_rate > 0.30:
                    current_scale *= 1.2  # Accept too often → larger steps
                elif current_rate < 0.15:
                    current_scale *= 0.8  # Accept too rarely → smaller steps
                # Clamp scale to prevent extreme values
                current_scale = np.clip(current_scale, 0.01, 1.0)
                warmup_accepts = 0
                warmup_iters = 0

            # Compute proposal Cholesky with current scale
            current_proposal_cov = current_scale ** 2 * omega
            try:
                current_proposal_chol = np.linalg.cholesky(current_proposal_cov)
            except np.linalg.LinAlgError:
                current_proposal_chol = proposal_chol  # fallback

            # Propose
            z = chain_rng.standard_normal(n_eta)
            eta_proposed = eta_current + current_proposal_chol @ z

            lp_proposed = _log_posterior(
                eta_proposed, tv_params, omega_inv,
                doses, observations, model.error_model, model_type,
            )

            # Accept/reject
            log_alpha = lp_proposed - lp_current
            if np.log(chain_rng.random()) < log_alpha:
                eta_current = eta_proposed
                lp_current = lp_proposed
                if it >= n_warmup:
                    n_accepted += 1
                else:
                    warmup_accepts += 1

            if it < n_warmup:
                warmup_iters += 1

            # Record post-warmup
            if it >= n_warmup:
                chain_samples.append(eta_current.copy())

        all_chains.append(np.array(chain_samples))

    # Combine chains
    combined = np.vstack(all_chains)
    total_samples = combined.shape[0]

    # Compute R-hat (Gelman-Rubin) if multiple chains
    rhat = 1.0
    if n_chains >= 2:
        chain_means = [c.mean(axis=0) for c in all_chains]
        chain_vars = [c.var(axis=0) for c in all_chains]
        overall_mean = combined.mean(axis=0)

        B = n_samples * np.mean([(m - overall_mean) ** 2 for m in chain_means], axis=0)
        W = np.mean(chain_vars, axis=0)

        var_hat = (1 - 1 / n_samples) * W + B / n_samples
        rhat_per_param = np.sqrt(var_hat / np.clip(W, 1e-20, None))
        rhat = float(np.max(rhat_per_param))

    # ESS
    ess_per_param = [_compute_ess(combined[:, i]) for i in range(n_eta)]
    ess = float(min(ess_per_param))

    # Acceptance rate (average across chains)
    acceptance_rate = n_accepted / n_samples  # Last chain

    # Convergence check
    converged = rhat < 1.1 and ess > 100

    # Best params: use posterior mean
    mean_eta = combined.mean(axis=0)
    map_params = apply_iiv(tv_params, mean_eta)

    # Summary statistics per parameter
    param_names = ["CL", "V1", "Q", "V2"][:n_eta]
    posterior_params = {}

    for i, name in enumerate(param_names):
        tv_val = getattr(tv_params, name)
        theta_samples = tv_val * np.exp(combined[:, i])
        posterior_params[name] = {
            "mean": float(np.mean(theta_samples)),
            "median": float(np.median(theta_samples)),
            "sd": float(np.std(theta_samples)),
            "ci95_lower": float(np.percentile(theta_samples, 2.5)),
            "ci95_upper": float(np.percentile(theta_samples, 97.5)),
        }

    return MCMCResultMH(
        posterior_eta=combined,
        posterior_params=posterior_params,
        map_params=map_params,
        acceptance_rate=acceptance_rate,
        n_samples=total_samples,
        converged=converged,
        rhat=rhat,
        ess=ess,
        n_divergences=total_samples - n_accepted,
    )
