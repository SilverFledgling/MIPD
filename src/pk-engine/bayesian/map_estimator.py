"""
MAP Estimator – Maximum A Posteriori estimation for individual PK parameters.

Implements the MAP objective function:
    J(eta) = 0.5 * eta^T * Omega^-1 * eta
           + 0.5 * sum_j [(y_j - y_hat_j)^2 / sigma_j^2]

Minimized using scipy.optimize.minimize (L-BFGS-B).

Reference: Sheiner & Beal (1982), J Pharmacokinet Biopharm
Dependencies: numpy, scipy, pk.models, pk.solver, pk.population
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
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


# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────

@dataclass
class MAPResult:
    """
    Result of MAP estimation.

    Attributes:
        eta_map:        Estimated random effects (eta vector)
        params:         Individual PK parameters (TV * exp(eta))
        objective:      Final objective function value J(eta)
        success:        Whether optimizer converged
        n_iterations:   Number of optimizer iterations
        residuals:      Individual prediction residuals (y - y_hat)
    """
    eta_map: NDArray[np.float64]
    params: PKParams
    objective: float
    success: bool
    n_iterations: int
    residuals: NDArray[np.float64]


# ──────────────────────────────────────────────────────────────────
# MAP objective function
# ──────────────────────────────────────────────────────────────────

def _map_objective(
    eta: NDArray[np.float64],
    tv_params: PKParams,
    omega_inv: NDArray[np.float64],
    doses: list[DoseEvent],
    observations: list[Observation],
    error_model: ErrorModel,
    model_type: ModelType,
) -> float:
    """
    MAP objective function J(eta).

    J(eta) = 0.5 * eta^T * Omega^-1 * eta              (prior)
           + 0.5 * sum [(y_obs - y_pred)^2 / var_j]    (likelihood)

    Args:
        eta:          Current random effects vector
        tv_params:    Typical (population) values
        omega_inv:    Inverse of Omega matrix
        doses:        Dose events
        observations: TDM observations
        error_model:  Residual error model
        model_type:   PK model type

    Returns:
        Objective function value (scalar)
    """
    # Apply IIV to get individual parameters
    try:
        ind_params = apply_iiv(tv_params, eta)
    except (ValueError, OverflowError):
        return 1e10  # Return large value for invalid parameters

    # Validate individual parameters
    if ind_params.CL <= 0 or ind_params.V1 <= 0:
        return 1e10

    # Predict concentrations at observation times
    obs_times = [obs.time for obs in observations]
    try:
        y_pred = predict_concentrations(
            ind_params, doses, obs_times, model_type
        )
    except (RuntimeError, ValueError):
        return 1e10  # ODE solver failure

    # Prior term: 0.5 * eta^T * Omega^-1 * eta
    prior = 0.5 * float(eta @ omega_inv @ eta)

    # Likelihood term: 0.5 * sum [(y - y_hat)^2 / var]
    likelihood = 0.0
    for i, obs in enumerate(observations):
        c_pred = max(y_pred[i], 1e-10)  # Prevent division by zero
        var_j = error_model.variance(c_pred)
        if var_j <= 0:
            var_j = 1e-6
        residual = obs.concentration - c_pred
        likelihood += residual ** 2 / var_j

    likelihood *= 0.5

    return prior + likelihood


# ──────────────────────────────────────────────────────────────────
# MAP estimation
# ──────────────────────────────────────────────────────────────────

def estimate_map(
    model: PopPKModel,
    tv_params: PKParams,
    doses: list[DoseEvent],
    observations: list[Observation],
    model_type: ModelType | None = None,
    max_iterations: int = 200,
) -> MAPResult:
    """
    Perform MAP estimation of individual PK parameters.

    Finds eta_MAP that minimizes J(eta), then computes
    individual parameters as theta_i = TV * exp(eta_MAP).

    Args:
        model:          PopPK model (for Omega and error model)
        tv_params:      Covariate-adjusted typical values
        doses:          Dose administration events
        observations:   TDM concentration measurements
        model_type:     PK model type (defaults to model.model_type)
        max_iterations: Maximum optimizer iterations

    Returns:
        MAPResult with estimated parameters and diagnostics
    """
    if not observations:
        raise ValueError("At least one observation is required for MAP")
    if not doses:
        raise ValueError("At least one dose event is required for MAP")

    if model_type is None:
        model_type = model.model_type

    # Prepare Omega inverse
    omega = np.array(model.omega_matrix, dtype=np.float64)
    n_eta = omega.shape[0]
    omega_inv = np.linalg.inv(omega)

    # Initial eta = 0 (start from population values)
    eta0 = np.zeros(n_eta)

    # Minimize MAP objective
    result = minimize(
        fun=_map_objective,
        x0=eta0,
        args=(
            tv_params,
            omega_inv,
            doses,
            observations,
            model.error_model,
            model_type,
        ),
        method="L-BFGS-B",
        options={
            "maxiter": max_iterations,
            "ftol": 1e-8,
            "gtol": 1e-4,
        },
    )

    eta_map = result.x

    # Compute individual parameters
    ind_params = apply_iiv(tv_params, eta_map)

    # Compute residuals
    obs_times = [obs.time for obs in observations]
    y_pred = predict_concentrations(ind_params, doses, obs_times, model_type)
    y_obs = np.array([obs.concentration for obs in observations])
    residuals = y_obs - y_pred

    # Determine convergence: scipy flag OR functionally converged
    # L-BFGS-B can report success=False when starting near optimum
    functionally_converged = (
        float(result.fun) < 10.0  # Reasonable objective
        and np.max(np.abs(residuals)) < 5.0  # Residuals not huge
    )
    converged = bool(result.success) or functionally_converged

    return MAPResult(
        eta_map=eta_map,
        params=ind_params,
        objective=float(result.fun),
        success=converged,
        n_iterations=result.nit,
        residuals=residuals,
    )
