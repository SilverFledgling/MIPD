"""
BNN Model – Bayesian Neural Network with MC Dropout for PK prediction.

Architecture:
    Input (covariates) → FC(64, ReLU, Dropout) → FC(32, ReLU, Dropout) → Output (PK param)

MC Dropout inference:
    At test time, keep dropout ON and run N forward passes.
    Mean of outputs = point estimate
    Std of outputs  = epistemic uncertainty

Advantages:
    - Captures non-linear covariate relationships
    - Provides calibrated uncertainty via MC sampling
    - Computationally cheaper than full BNN (VI/HMC)

Reference:
    - Gal & Ghahramani (2016), ICML – Dropout as Bayesian Approximation
    - Sibieude et al. (2021), CPT Pharmacometrics

Dependencies: numpy, scipy (no PyTorch/TF required – pure numpy implementation)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# ──────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────

@dataclass
class BNNPrediction:
    """
    BNN prediction with uncertainty.

    Attributes:
        mean:       Point estimate (mean of MC samples)
        std:        Epistemic uncertainty (std of MC samples)
        ci95_lower: 95% CI lower bound
        ci95_upper: 95% CI upper bound
        mc_samples: All MC forward pass outputs
    """
    mean: float
    std: float
    ci95_lower: float
    ci95_upper: float
    mc_samples: NDArray[np.float64]


@dataclass
class BNNModel:
    """
    Trained BNN model.

    Attributes:
        parameter_name: Target PK parameter
        feature_names:  Input covariate names
        weights:        List of weight matrices per layer
        biases:         List of bias vectors per layer
        dropout_rate:   Dropout probability
        x_mean:         Feature means
        x_std:          Feature stds
        y_mean:         Target mean
        y_std:          Target std
    """
    parameter_name: str
    feature_names: list[str]
    weights: list[NDArray[np.float64]]
    biases: list[NDArray[np.float64]]
    dropout_rate: float
    x_mean: NDArray[np.float64]
    x_std: NDArray[np.float64]
    y_mean: float
    y_std: float


# ──────────────────────────────────────────────────────────────────
# Activation functions
# ──────────────────────────────────────────────────────────────────

def _relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """ReLU activation: max(0, x)."""
    return np.maximum(0.0, x)


def _relu_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """ReLU derivative: 1 if x > 0 else 0."""
    return (x > 0).astype(np.float64)


# ──────────────────────────────────────────────────────────────────
# Forward pass
# ──────────────────────────────────────────────────────────────────

def _forward(
    x: NDArray[np.float64],
    weights: list[NDArray[np.float64]],
    biases: list[NDArray[np.float64]],
    dropout_rate: float,
    rng: np.random.Generator,
    training: bool = True,
) -> NDArray[np.float64]:
    """
    Forward pass through the BNN.

    Args:
        x:            Input (batch_size, n_features)
        weights:      Weight matrices
        biases:       Bias vectors
        dropout_rate: Dropout probability
        rng:          Random number generator
        training:     If True, apply dropout (MC mode)

    Returns:
        Output predictions (batch_size, 1)
    """
    h = x
    n_layers = len(weights)

    for i in range(n_layers):
        h = h @ weights[i] + biases[i]

        # Apply ReLU for hidden layers (not output)
        if i < n_layers - 1:
            h = _relu(h)

            # Apply dropout (during training AND MC inference)
            if training and dropout_rate > 0:
                mask = rng.binomial(1, 1 - dropout_rate, h.shape)
                h = h * mask / (1 - dropout_rate)  # Inverted dropout

    return h


# ──────────────────────────────────────────────────────────────────
# Training (mini-batch SGD with L2 regularization)
# ──────────────────────────────────────────────────────────────────

def train_bnn(
    X_raw: NDArray[np.float64],
    y_raw: NDArray[np.float64],
    parameter_name: str = "CL",
    feature_names: list[str] | None = None,
    hidden_sizes: tuple[int, ...] = (64, 32),
    dropout_rate: float = 0.20,
    learning_rate: float = 0.001,
    n_epochs: int = 500,
    batch_size: int = 32,
    l2_lambda: float = 1e-4,
    seed: int = 42,
) -> BNNModel:
    """
    Train a BNN with MC Dropout.

    Args:
        X_raw:           (n, d) covariate matrix
        y_raw:           (n,) PK parameter values
        parameter_name:  Name of PK parameter
        feature_names:   Covariate names
        hidden_sizes:    Hidden layer sizes
        dropout_rate:    Dropout probability
        learning_rate:   SGD learning rate
        n_epochs:        Training epochs
        batch_size:      Mini-batch size
        l2_lambda:       L2 regularization strength
        seed:            Random seed

    Returns:
        Trained BNNModel
    """
    rng = np.random.default_rng(seed)
    n, d = X_raw.shape

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(d)]

    # Standardize
    x_mean = np.mean(X_raw, axis=0)
    x_std = np.std(X_raw, axis=0)
    x_std = np.where(x_std < 1e-10, 1.0, x_std)

    y_mean = float(np.mean(y_raw))
    y_std = float(np.std(y_raw))
    if y_std < 1e-10:
        y_std = 1.0

    X = (X_raw - x_mean) / x_std
    y = ((y_raw - y_mean) / y_std).reshape(-1, 1)

    # Initialize weights (He initialization)
    layer_sizes = [d] + list(hidden_sizes) + [1]
    weights: list[NDArray[np.float64]] = []
    biases: list[NDArray[np.float64]] = []

    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        std_init = np.sqrt(2.0 / fan_in)
        W = rng.normal(0, std_init, (fan_in, fan_out))
        b = np.zeros((1, fan_out))
        weights.append(W)
        biases.append(b)

    # Training loop (mini-batch SGD with backprop)
    for epoch in range(n_epochs):
        # Shuffle data
        indices = rng.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            m = X_batch.shape[0]

            # Forward pass with cache for backprop
            activations = [X_batch]
            pre_activations = []
            h = X_batch

            for i in range(len(weights)):
                z = h @ weights[i] + biases[i]
                pre_activations.append(z)

                if i < len(weights) - 1:
                    h = _relu(z)
                    # Dropout during training
                    if dropout_rate > 0:
                        mask = rng.binomial(
                            1, 1 - dropout_rate, h.shape,
                        ).astype(np.float64)
                        h = h * mask / (1 - dropout_rate)
                else:
                    h = z  # Linear output

                activations.append(h)

            # Loss: MSE + L2
            y_pred = activations[-1]
            loss_grad = 2.0 * (y_pred - y_batch) / m  # dL/dy_pred

            # Backpropagation
            delta = loss_grad
            for i in range(len(weights) - 1, -1, -1):
                # Gradient for weights and biases
                dW = activations[i].T @ delta + 2 * l2_lambda * weights[i]
                db = np.sum(delta, axis=0, keepdims=True)

                # Propagate gradient
                if i > 0:
                    delta = delta @ weights[i].T
                    delta *= _relu_derivative(pre_activations[i - 1])

                # Update
                weights[i] -= learning_rate * dW
                biases[i] -= learning_rate * db

    return BNNModel(
        parameter_name=parameter_name,
        feature_names=feature_names,
        weights=weights,
        biases=biases,
        dropout_rate=dropout_rate,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )


# ──────────────────────────────────────────────────────────────────
# MC Dropout prediction
# ──────────────────────────────────────────────────────────────────

def predict_bnn(
    model: BNNModel,
    X_new: NDArray[np.float64],
    n_mc_samples: int = 100,
    seed: int = 42,
) -> list[BNNPrediction]:
    """
    MC Dropout prediction with uncertainty.

    Runs n_mc_samples stochastic forward passes with dropout ON,
    then computes mean (point estimate) and std (uncertainty).

    Args:
        model:          Trained BNN
        X_new:          (n_new, d) covariate matrix
        n_mc_samples:   Number of MC forward passes
        seed:           Random seed

    Returns:
        List of BNNPrediction (one per query point)
    """
    rng = np.random.default_rng(seed)

    # Standardize
    X_test = (X_new - model.x_mean) / model.x_std

    # MC forward passes
    all_preds = np.zeros((n_mc_samples, X_test.shape[0]))
    for t in range(n_mc_samples):
        y_std = _forward(
            X_test, model.weights, model.biases,
            model.dropout_rate, rng, training=True,
        )
        # De-standardize
        all_preds[t, :] = (y_std.flatten() * model.y_std + model.y_mean)

    # Compute statistics
    predictions = []
    for i in range(X_test.shape[0]):
        samples = all_preds[:, i]
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=1))

        predictions.append(BNNPrediction(
            mean=mean,
            std=std,
            ci95_lower=float(np.percentile(samples, 2.5)),
            ci95_upper=float(np.percentile(samples, 97.5)),
            mc_samples=samples,
        ))

    return predictions
