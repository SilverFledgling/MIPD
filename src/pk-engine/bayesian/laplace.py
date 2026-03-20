"""
Laplace Approximation – Gaussian approximation of posterior distribution.

After MAP estimation, computes:
    Sigma_post = [-d^2 log p(theta|y) / d theta^2]^(-1)
    CI_95 = theta_MAP +/- 1.96 * sqrt(diag(Sigma_post))

Also supports online Bayesian updating when new TDM data arrives:
    Sigma_new^-1 = Sigma_old^-1 + J^T * Sigma_eps^-1 * J
    theta_new = theta_old + Sigma_new * J^T * Sigma_eps^-1 * (y_new - y_hat)

Reference: Tierney & Kadane (1986), JASA
Dependencies: numpy, scipy, pk.models, bayesian.map_estimator
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import approx_fprime
from dataclasses import dataclass

from pk.models import (
    DoseEvent,
    ErrorModel,
    ModelType,
    Observation,
    PKParams,
    PopPKModel,
)
from pk.solver import predict_concentrations
from pk.population import apply_iiv
from bayesian.map_estimator import MAPResult, _map_objective


# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────

@dataclass
class LaplaceResult:
    """
    Result of Laplace approximation.

    Attributes:
        map_result:       Underlying MAP estimation result
        posterior_cov:    Posterior covariance matrix Sigma_post
        ci_lower:         Lower 95% credible interval for PKParams
        ci_upper:         Upper 95% credible interval for PKParams
        param_names:      Names of parameters
        param_values:     MAP parameter values
        param_sds:        Posterior standard deviations
    """
    map_result: MAPResult
    posterior_cov: NDArray[np.float64]
    ci_lower: dict[str, float]
    ci_upper: dict[str, float]
    param_names: list[str]
    param_values: list[float]
    param_sds: list[float]


# ──────────────────────────────────────────────────────────────────
# Compute Hessian via finite differences
# ──────────────────────────────────────────────────────────────────

def _compute_hessian(
    eta: NDArray[np.float64],
    tv_params: PKParams,
    omega_inv: NDArray[np.float64],
    doses: list[DoseEvent],
    observations: list[Observation],
    error_model: ErrorModel,
    model_type: ModelType,
    epsilon: float = 1e-3,
) -> NDArray[np.float64]:
    """
    Compute Hessian of MAP objective via central finite differences.

    Uses direct second-order central difference formula:
        H_ii = [f(x+h_i) - 2f(x) + f(x-h_i)] / h^2
        H_ij = [f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j)] / (4h^2)

    Epsilon=1e-3 (larger than typical 1e-5) for PK models where
    the objective landscape is relatively smooth.

    Args:
        eta:       Current eta values (at MAP)
        Other args: same as _map_objective

    Returns:
        Hessian matrix (n_eta x n_eta)
    """
    n = len(eta)
    hessian = np.zeros((n, n))

    def obj(e: NDArray[np.float64]) -> float:
        return _map_objective(
            e, tv_params, omega_inv, doses, observations,
            error_model, model_type,
        )

    f0 = obj(eta)

    # Diagonal elements: d²f/dx_i²
    for i in range(n):
        e_plus = eta.copy()
        e_minus = eta.copy()
        e_plus[i] += epsilon
        e_minus[i] -= epsilon
        hessian[i, i] = (obj(e_plus) - 2.0 * f0 + obj(e_minus)) / (epsilon ** 2)

    # Off-diagonal: d²f/(dx_i dx_j) via cross differences
    for i in range(n):
        for j in range(i + 1, n):
            e_pp = eta.copy()  # +i, +j
            e_pm = eta.copy()  # +i, -j
            e_mp = eta.copy()  # -i, +j
            e_mm = eta.copy()  # -i, -j

            e_pp[i] += epsilon; e_pp[j] += epsilon
            e_pm[i] += epsilon; e_pm[j] -= epsilon
            e_mp[i] -= epsilon; e_mp[j] += epsilon
            e_mm[i] -= epsilon; e_mm[j] -= epsilon

            h_ij = (obj(e_pp) - obj(e_pm) - obj(e_mp) + obj(e_mm)) / (4.0 * epsilon ** 2)
            hessian[i, j] = h_ij
            hessian[j, i] = h_ij

    # Ensure Hessian is positive definite (it should be at MAP minimum)
    eigvals, eigvecs = np.linalg.eigh(hessian)
    eigvals = np.maximum(eigvals, 1e-4)  # Floor at small positive value
    hessian = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return hessian


# ──────────────────────────────────────────────────────────────────
# Laplace approximation
# ──────────────────────────────────────────────────────────────────

_PARAM_NAMES_4 = ["CL", "V1", "Q", "V2"]
_PARAM_NAMES_2 = ["CL", "V1"]


def laplace_approximation(
    map_result: MAPResult,
    model: PopPKModel,
    tv_params: PKParams,
    doses: list[DoseEvent],
    observations: list[Observation],
    model_type: ModelType | None = None,
) -> LaplaceResult:
    """
    Compute Laplace approximation of posterior after MAP estimation.

    Steps:
        1. Compute Hessian H at eta_MAP
        2. Sigma_post = H^(-1)
        3. CI = theta_MAP +/- 1.96 * sqrt(diag(Sigma_post on theta scale))

    Args:
        map_result:   Result from MAP estimation
        model:        PopPK model definition
        tv_params:    Covariate-adjusted typical values
        doses:        Dose events
        observations: TDM observations
        model_type:   PK model type

    Returns:
        LaplaceResult with posterior covariance and credible intervals
    """
    if model_type is None:
        model_type = model.model_type

    omega = np.array(model.omega_matrix, dtype=np.float64)
    omega_inv = np.linalg.inv(omega)
    n_eta = len(map_result.eta_map)

    # Compute Hessian at MAP estimate
    hessian = _compute_hessian(
        eta=map_result.eta_map,
        tv_params=tv_params,
        omega_inv=omega_inv,
        doses=doses,
        observations=observations,
        error_model=model.error_model,
        model_type=model_type,
    )

    # Posterior covariance = inverse of Hessian
    try:
        posterior_cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        # If Hessian is singular, use pseudo-inverse
        posterior_cov = np.linalg.pinv(hessian)

    # Ensure covariance is positive semi-definite
    posterior_cov = _ensure_positive_definite(posterior_cov)

    # Parameter names
    param_names = _PARAM_NAMES_4[:n_eta] if n_eta <= 4 else [
        f"param_{i}" for i in range(n_eta)
    ]

    # Get MAP parameter values
    p = map_result.params
    param_values_map = [p.CL, p.V1]
    if n_eta >= 3:
        param_values_map.append(p.Q)
    if n_eta >= 4:
        param_values_map.append(p.V2)

    # Compute CIs on the original (theta) scale
    # Since theta = TV * exp(eta), the SD on theta scale involves
    # the Jacobian of the transformation
    eta_sds = np.sqrt(np.abs(np.diag(posterior_cov)))

    ci_lower: dict[str, float] = {}
    ci_upper: dict[str, float] = {}
    param_sds: list[float] = []

    tv_values = [tv_params.CL, tv_params.V1]
    if n_eta >= 3:
        tv_values.append(tv_params.Q)
    if n_eta >= 4:
        tv_values.append(tv_params.V2)

    for i, name in enumerate(param_names):
        # On eta scale: CI for eta_i
        eta_i = map_result.eta_map[i]
        eta_lo = eta_i - 1.96 * eta_sds[i]
        eta_hi = eta_i + 1.96 * eta_sds[i]

        # Transform back to theta scale: theta = TV * exp(eta)
        tv_i = tv_values[i]
        theta_lo = tv_i * np.exp(eta_lo)
        theta_hi = tv_i * np.exp(eta_hi)

        ci_lower[name] = float(theta_lo)
        ci_upper[name] = float(theta_hi)

        # Approximate SD on theta scale via delta method
        theta_sd = param_values_map[i] * eta_sds[i]
        param_sds.append(float(theta_sd))

    return LaplaceResult(
        map_result=map_result,
        posterior_cov=posterior_cov,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        param_names=param_names,
        param_values=param_values_map,
        param_sds=param_sds,
    )


def _ensure_positive_definite(
    matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Ensure matrix is positive definite by adjusting eigenvalues."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    min_eigenvalue = 1e-8
    eigvals = np.maximum(eigvals, min_eigenvalue)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ──────────────────────────────────────────────────────────────────
# Online Bayesian update (adaptive)
# ──────────────────────────────────────────────────────────────────

def bayesian_update(
    theta_old: NDArray[np.float64],
    sigma_old_inv: NDArray[np.float64],
    y_new: float,
    y_pred_new: float,
    jacobian: NDArray[np.float64],
    sigma_eps: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Online Bayesian update with new TDM observation.

    Sigma_new^-1 = Sigma_old^-1 + J^T * (1/sigma_eps^2) * J
    theta_new = theta_old + Sigma_new * J^T * (1/sigma_eps^2) * (y_new - y_hat)

    Args:
        theta_old:     Previous MAP estimate
        sigma_old_inv: Inverse of previous posterior covariance
        y_new:         New observed concentration
        y_pred_new:    Predicted concentration at new obs
        jacobian:      Jacobian d(y_pred)/d(theta) at theta_old
        sigma_eps:     Residual error SD

    Returns:
        Tuple of (theta_new, sigma_new_inv)
    """
    j = jacobian.reshape(-1, 1)  # Column vector
    precision_eps = 1.0 / (sigma_eps ** 2)

    # Update precision matrix
    sigma_new_inv = sigma_old_inv + precision_eps * (j @ j.T)

    # Update theta
    sigma_new = np.linalg.inv(sigma_new_inv)
    innovation = y_new - y_pred_new
    theta_new = theta_old + (
        sigma_new @ j * precision_eps * innovation
    ).flatten()

    return theta_new, sigma_new_inv
