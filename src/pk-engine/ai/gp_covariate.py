"""
GP Covariate Model – Gaussian Process for non-linear covariate-PK relationships.

Instead of hardcoded allometric equations, a GP learns the mapping:
    f: (CrCL, Weight, Age, Albumin, ...) -> PK parameter (CL, V1, ...)

Advantages over parametric covariate models:
    - Captures non-linear, non-monotonic relationships
    - Provides uncertainty quantification (posterior variance)
    - Automatically discovers interactions between covariates

Kernel: RBF (Squared Exponential) + White noise
    k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * l^2)) + sigma_n^2 * I

Reference:
    - Rasmussen & Williams (2006), Gaussian Processes for ML
    - Ribba et al. (2014), CPT Pharmacometrics Syst Pharmacol

Dependencies: numpy, scipy (no GPyTorch/PyTorch required)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


# ──────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────

@dataclass
class GPPrediction:
    """
    GP prediction result for a single query point.

    Attributes:
        mean:       Predicted PK parameter value
        std:        Predictive standard deviation
        ci95_lower: 95% CI lower bound
        ci95_upper: 95% CI upper bound
    """
    mean: float
    std: float
    ci95_lower: float
    ci95_upper: float


@dataclass
class GPCovariateModel:
    """
    Trained GP model for one PK parameter.

    Attributes:
        parameter_name: Target PK parameter (e.g., "CL")
        feature_names:  Covariate names
        X_train:        Training covariates (n_train, n_features)
        y_train:        Training PK values (n_train,)
        length_scale:   RBF length scale per feature
        signal_var:     Signal variance (sigma_f^2)
        noise_var:      Noise variance (sigma_n^2)
        alpha:          Precomputed K^-1 * y (for fast prediction)
        L:              Cholesky factor of K + noise * I
        x_mean:         Feature means (for standardization)
        x_std:          Feature stds (for standardization)
        y_mean:         Target mean
        y_std:          Target std
    """
    parameter_name: str
    feature_names: list[str]
    X_train: NDArray[np.float64]
    y_train: NDArray[np.float64]
    length_scale: NDArray[np.float64]
    signal_var: float
    noise_var: float
    alpha: NDArray[np.float64]
    L: NDArray[np.float64]
    L_lower: bool
    x_mean: NDArray[np.float64]
    x_std: NDArray[np.float64]
    y_mean: float
    y_std: float


# ──────────────────────────────────────────────────────────────────
# Kernel functions
# ──────────────────────────────────────────────────────────────────

def rbf_kernel(
    X1: NDArray[np.float64],
    X2: NDArray[np.float64],
    length_scale: NDArray[np.float64],
    signal_var: float,
) -> NDArray[np.float64]:
    """
    Anisotropic RBF (Squared Exponential) kernel.

    k(x, x') = sigma_f^2 * exp(-0.5 * sum_d ((x_d - x'_d) / l_d)^2)

    Args:
        X1: (n1, d) feature matrix
        X2: (n2, d) feature matrix
        length_scale: (d,) per-feature length scales
        signal_var: Signal variance sigma_f^2

    Returns:
        (n1, n2) kernel matrix
    """
    # Scale features by length scale
    X1_scaled = X1 / length_scale
    X2_scaled = X2 / length_scale

    # Squared Euclidean distance
    sq_dist = (
        np.sum(X1_scaled ** 2, axis=1, keepdims=True)
        + np.sum(X2_scaled ** 2, axis=1, keepdims=True).T
        - 2.0 * X1_scaled @ X2_scaled.T
    )

    return signal_var * np.exp(-0.5 * sq_dist)


# ──────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────

def _neg_log_marginal_likelihood(
    theta: NDArray[np.float64],
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """
    Negative log marginal likelihood for hyperparameter optimization.

    log p(y|X, theta) = -0.5 * y^T K^-1 y - 0.5 * log|K| - n/2 * log(2pi)
    """
    n = X.shape[0]
    d = X.shape[1]

    # Unpack hyperparameters (log-space for positivity)
    log_length_scales = theta[:d]
    log_signal_var = theta[d]
    log_noise_var = theta[d + 1]

    length_scale = np.exp(log_length_scales)
    signal_var = np.exp(log_signal_var)
    noise_var = np.exp(log_noise_var)

    K = rbf_kernel(X, X, length_scale, signal_var)
    K += noise_var * np.eye(n) + 1e-8 * np.eye(n)  # Jitter

    try:
        L, lower = cho_factor(K)
        alpha = cho_solve((L, lower), y)

        data_fit = 0.5 * float(y @ alpha)
        complexity = float(np.sum(np.log(np.diag(L))))
        constant = 0.5 * n * np.log(2 * np.pi)

        return data_fit + complexity + constant
    except np.linalg.LinAlgError:
        return 1e10  # Non-PD matrix


def train_gp(
    X_raw: NDArray[np.float64],
    y_raw: NDArray[np.float64],
    parameter_name: str = "CL",
    feature_names: list[str] | None = None,
    optimize_hyperparams: bool = True,
) -> GPCovariateModel:
    """
    Train a GP covariate model.

    Steps:
        1. Standardize features and target
        2. Optimize hyperparameters (length scale, signal/noise var)
        3. Precompute K^-1 * y for fast prediction

    Args:
        X_raw:              (n, d) covariate matrix
        y_raw:              (n,) PK parameter values
        parameter_name:     Name of PK parameter
        feature_names:      Covariate names
        optimize_hyperparams: Whether to optimize hyperparameters

    Returns:
        Trained GPCovariateModel
    """
    n, d = X_raw.shape

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(d)]

    # Standardize
    x_mean = np.mean(X_raw, axis=0)
    x_std = np.std(X_raw, axis=0)
    x_std = np.where(x_std < 1e-10, 1.0, x_std)  # Avoid division by zero

    y_mean = float(np.mean(y_raw))
    y_std = float(np.std(y_raw))
    if y_std < 1e-10:
        y_std = 1.0

    X = (X_raw - x_mean) / x_std
    y = (y_raw - y_mean) / y_std

    # Initial hyperparameters (log-space)
    log_length_scales = np.zeros(d)        # l = 1.0
    log_signal_var = np.log(1.0)           # sigma_f = 1.0
    log_noise_var = np.log(0.1)            # sigma_n = sqrt(0.1)

    theta0 = np.concatenate([
        log_length_scales,
        [log_signal_var, log_noise_var],
    ])

    if optimize_hyperparams and n >= 5:
        result = minimize(
            _neg_log_marginal_likelihood,
            theta0,
            args=(X, y),
            method="L-BFGS-B",
            options={"maxiter": 100},
        )
        theta_opt = result.x
    else:
        theta_opt = theta0

    length_scale = np.exp(theta_opt[:d])
    signal_var = float(np.exp(theta_opt[d]))
    noise_var = float(np.exp(theta_opt[d + 1]))

    # Compute kernel matrix and precompute alpha
    K = rbf_kernel(X, X, length_scale, signal_var)
    K += noise_var * np.eye(n) + 1e-8 * np.eye(n)

    L, lower = cho_factor(K)
    alpha = cho_solve((L, lower), y)

    return GPCovariateModel(
        parameter_name=parameter_name,
        feature_names=feature_names,
        X_train=X,
        y_train=y,
        length_scale=length_scale,
        signal_var=signal_var,
        noise_var=noise_var,
        alpha=alpha,
        L=L,
        L_lower=lower,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )


# ──────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────

def predict_gp(
    model: GPCovariateModel,
    X_new: NDArray[np.float64],
) -> list[GPPrediction]:
    """
    Predict PK parameter values for new covariate vectors.

    Uses GP posterior mean and variance:
        mu* = K(X*, X) @ alpha
        var* = K(X*, X*) - K(X*, X) @ K^-1 @ K(X, X*)

    Args:
        model: Trained GP model
        X_new: (n_new, d) new covariate matrix

    Returns:
        List of GPPrediction (one per query point)
    """
    # Standardize
    X_test = (X_new - model.x_mean) / model.x_std

    # K(X*, X_train)
    K_star = rbf_kernel(
        X_test, model.X_train, model.length_scale, model.signal_var,
    )

    # Posterior mean (standardized)
    mu_std = K_star @ model.alpha

    # Posterior variance
    K_ss = rbf_kernel(
        X_test, X_test, model.length_scale, model.signal_var,
    )
    v = cho_solve((model.L, model.L_lower), K_star.T)
    var_std = np.diag(K_ss) - np.sum(K_star.T * v, axis=0)
    var_std = np.maximum(var_std, 0.0)  # Clip negative variance

    # De-standardize
    mu = mu_std * model.y_std + model.y_mean
    std = np.sqrt(var_std) * model.y_std

    predictions = []
    for i in range(len(mu)):
        predictions.append(GPPrediction(
            mean=float(mu[i]),
            std=float(std[i]),
            ci95_lower=float(mu[i] - 1.96 * std[i]),
            ci95_upper=float(mu[i] + 1.96 * std[i]),
        ))

    return predictions
