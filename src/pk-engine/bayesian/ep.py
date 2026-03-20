"""
EP – Expectation Propagation for PK parameter estimation.

Approximates posterior by iteratively refining local site approximations:
    p(eta|y) ≈ q(eta) = prod_j t_j(eta) * p(eta)

Each site t_j corresponds to one observation likelihood factor.
Global approximation maintained as a Gaussian.

Reference:
    - Minka (2001), Expectation Propagation for Approximate Bayesian Inference
    - Vehtari et al. (2020), Bayesian Analysis 15(2)

Dependencies: numpy, scipy, pk.models, pk.solver, pk.population
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
class EPResult:
    """
    Result of Expectation Propagation.

    Attributes:
        mu:          Posterior mean (eta space)
        cov:         Posterior covariance
        params:      PK parameters at posterior mean
        n_iterations: EP iterations to convergence
        converged:   Whether EP converged
    """
    mu: NDArray[np.float64]
    cov: NDArray[np.float64]
    params: PKParams
    n_iterations: int
    converged: bool


def _compute_log_likelihood_at_eta(
    eta: NDArray[np.float64],
    tv_params: PKParams,
    doses: list[DoseEvent],
    obs: Observation,
    error_model: ErrorModel,
    model_type: ModelType,
) -> float:
    """Log-likelihood for a single observation at given eta."""
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


def run_ep(
    model: PopPKModel,
    tv_params: PKParams,
    doses: list[DoseEvent],
    observations: list[Observation],
    model_type: ModelType | None = None,
    max_iterations: int = 20,
    tol: float = 1e-4,
    n_quadrature: int = 5,
) -> EPResult:
    """
    Run Expectation Propagation.

    Args:
        model:           PopPK model
        tv_params:       Typical values
        doses:           Dose events
        observations:    TDM observations
        model_type:      PK model type
        max_iterations:  Max EP sweeps
        tol:             Convergence tolerance
        n_quadrature:    Quadrature points for moment matching

    Returns:
        EPResult with posterior mean and covariance
    """
    if not observations:
        raise ValueError("At least one observation required")
    if not doses:
        raise ValueError("At least one dose required")

    if model_type is None:
        model_type = model.model_type

    omega = np.array(model.omega_matrix, dtype=np.float64)
    n_eta = omega.shape[0]
    n_obs = len(observations)

    # Prior: N(0, Omega)
    prior_prec = np.linalg.inv(omega)
    prior_nat_mean = np.zeros(n_eta)  # Lambda * mu = 0

    # Initialize site approximations (natural parameters)
    site_prec = [np.zeros((n_eta, n_eta)) for _ in range(n_obs)]
    site_nat_mean = [np.zeros(n_eta) for _ in range(n_obs)]

    converged = False
    n_iter = 0

    for iteration in range(max_iterations):
        n_iter = iteration + 1
        max_change = 0.0

        for j in range(n_obs):
            # Cavity distribution: remove site j
            cavity_prec = prior_prec.copy()
            cavity_nat_mean = prior_nat_mean.copy()
            for k in range(n_obs):
                if k != j:
                    cavity_prec += site_prec[k]
                    cavity_nat_mean += site_nat_mean[k]

            # Ensure PD
            try:
                cavity_cov = np.linalg.inv(cavity_prec)
            except np.linalg.LinAlgError:
                continue

            cavity_mu = cavity_cov @ cavity_nat_mean

            # Moment matching via sigma-point approximation
            # Sample around cavity mean, weight by likelihood
            # Ensure cavity_cov is positive definite via eigenvalue clipping
            eigvals, eigvecs = np.linalg.eigh(cavity_cov)
            eigvals = np.maximum(eigvals, 1e-6)  # Clip negative eigenvalues
            cavity_cov_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L_cav = np.linalg.cholesky(cavity_cov_pd)

            # Sigma points (simplified unscented)
            sigma_points = [cavity_mu]
            scale = np.sqrt(n_eta)
            for i in range(n_eta):
                sigma_points.append(cavity_mu + scale * L_cav[:, i])
                sigma_points.append(cavity_mu - scale * L_cav[:, i])

            weights = np.ones(len(sigma_points)) / len(sigma_points)

            # Compute tilted moments
            log_likes = []
            for sp in sigma_points:
                ll = _compute_log_likelihood_at_eta(
                    sp, tv_params, doses, observations[j],
                    model.error_model, model_type,
                )
                log_likes.append(ll)

            log_likes = np.array(log_likes)
            max_ll = np.max(log_likes)
            likes = np.exp(log_likes - max_ll)
            weighted_likes = weights * likes
            Z = np.sum(weighted_likes)

            if Z < 1e-30:
                continue

            # Tilted mean and covariance
            w_norm = weighted_likes / Z
            tilted_mu = sum(
                w_norm[i] * sigma_points[i]
                for i in range(len(sigma_points))
            )
            tilted_cov = np.zeros((n_eta, n_eta))
            for i in range(len(sigma_points)):
                diff = sigma_points[i] - tilted_mu
                tilted_cov += w_norm[i] * np.outer(diff, diff)
            tilted_cov += 1e-8 * np.eye(n_eta)

            # Update site
            try:
                tilted_prec = np.linalg.inv(tilted_cov)
            except np.linalg.LinAlgError:
                continue

            new_site_prec = tilted_prec - cavity_prec
            new_site_nat_mean = tilted_prec @ tilted_mu - cavity_nat_mean

            # Check positive-semidefiniteness of site precision
            # If eigenvalues go negative, the EP update is unstable → skip
            site_eigs = np.linalg.eigvalsh(new_site_prec)
            if np.any(site_eigs < -1.0):
                # Site precision too negative → skip this update
                continue

            # Damping for stability (higher damping = more conservative)
            damping = 0.3  # Reduced from 0.5 for better stability
            new_site_prec = (
                (1 - damping) * site_prec[j] + damping * new_site_prec
            )
            new_site_nat_mean = (
                (1 - damping) * site_nat_mean[j] + damping * new_site_nat_mean
            )

            # Check resulting site precision is not too extreme
            max_prec_val = np.max(np.abs(new_site_prec))
            if max_prec_val > 1e6 or np.any(np.isnan(new_site_prec)):
                continue

            change = float(np.max(np.abs(new_site_prec - site_prec[j])))
            max_change = max(max_change, change)

            site_prec[j] = new_site_prec
            site_nat_mean[j] = new_site_nat_mean

        if max_change < tol:
            converged = True
            break

    # Final posterior
    post_prec = prior_prec.copy()
    post_nat_mean = prior_nat_mean.copy()
    for j in range(n_obs):
        post_prec += site_prec[j]
        post_nat_mean += site_nat_mean[j]

    # Validate posterior precision is positive definite
    try:
        post_prec_eigs = np.linalg.eigvalsh(post_prec)
        if np.any(post_prec_eigs <= 0):
            # Posterior precision not PD → EP failed, fallback to prior
            post_cov = omega.copy()
            post_mu = np.zeros(n_eta)
            converged = False
        else:
            post_cov = np.linalg.inv(post_prec)
            # Ensure positive definite
            eigvals, eigvecs = np.linalg.eigh(post_cov)
            eigvals = np.maximum(eigvals, 1e-8)
            post_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            post_mu = post_cov @ post_nat_mean
    except np.linalg.LinAlgError:
        post_cov = omega.copy()
        post_mu = np.zeros(n_eta)
        converged = False

    # Clamp eta to prevent overflow — exp(eta) must stay physiological
    # ±2.0 corresponds to ~7× or ~0.14× population value (safer than ±3.0)
    post_mu = np.clip(post_mu, -2.0, 2.0)

    # Sanity check: if CL or V1 would be extreme, fallback to zero eta
    try:
        test_params = apply_iiv(tv_params, post_mu)
        if test_params.CL <= 0 or test_params.V1 <= 0 or test_params.CL > 50 or test_params.V1 > 500:
            post_mu = np.zeros(n_eta)
            post_cov = omega.copy()
            converged = False
    except (ValueError, OverflowError):
        post_mu = np.zeros(n_eta)
        post_cov = omega.copy()
        converged = False

    params = apply_iiv(tv_params, post_mu)

    return EPResult(
        mu=post_mu,
        cov=post_cov,
        params=params,
        n_iterations=n_iter,
        converged=converged,
    )

