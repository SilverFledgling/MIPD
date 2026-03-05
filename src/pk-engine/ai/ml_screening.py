"""
ML Screening – Random Forest, Neural Network, SVR for covariate screening.

Identifies which covariates most influence PK parameters using
multiple ML methods, then ranks them by importance.

Protocol (Sibieude et al. 2021):
    1. Train RF, NN, SVR on covariates -> PK parameter
    2. Compute feature importance for each model
    3. Aggregate rankings via Borda count
    4. Select top-K covariates for inclusion in PopPK model

Reference:
    - Sibieude et al. (2021), CPT Pharmacometrics Syst Pharmacol
    - Breiman (2001), Random Forests, Machine Learning 45(1)

Dependencies: numpy, scipy (pure implementation, no sklearn required)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────

@dataclass
class ScreeningResult:
    """
    Result of covariate screening.

    Attributes:
        feature_names:     Covariate names
        rf_importance:     Random Forest feature importance
        nn_importance:     Neural Network permutation importance
        svr_importance:    SVR permutation importance
        borda_scores:      Aggregated Borda count scores
        ranking:           Final covariate ranking (best first)
        selected:          Top-K selected covariates
    """
    feature_names: list[str]
    rf_importance: NDArray[np.float64]
    nn_importance: NDArray[np.float64]
    svr_importance: NDArray[np.float64]
    borda_scores: NDArray[np.float64]
    ranking: list[str]
    selected: list[str]


# ──────────────────────────────────────────────────────────────────
# Random Forest (simplified: ensemble of decision stumps)
# ──────────────────────────────────────────────────────────────────

class _DecisionStump:
    """Single-split decision tree (depth=1)."""

    def __init__(self) -> None:
        self.feature_idx: int = 0
        self.threshold: float = 0.0
        self.left_val: float = 0.0
        self.right_val: float = 0.0
        self.importance: float = 0.0

    def fit(
        self, X: NDArray, y: NDArray, rng: np.random.Generator,
    ) -> None:
        """Find best split among random subset of features."""
        n, d = X.shape
        n_try = max(1, int(np.sqrt(d)))
        features = rng.choice(d, n_try, replace=False)

        best_mse = float("inf")
        total_var = float(np.var(y))

        for feat in features:
            vals = np.unique(X[:, feat])
            if len(vals) < 2:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2
            # Sample at most 10 thresholds
            if len(thresholds) > 10:
                thresholds = rng.choice(thresholds, 10, replace=False)

            for thr in thresholds:
                left = y[X[:, feat] <= thr]
                right = y[X[:, feat] > thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                mse = (len(left) * np.var(left) + len(right) * np.var(right)) / n
                if mse < best_mse:
                    best_mse = mse
                    self.feature_idx = feat
                    self.threshold = float(thr)
                    self.left_val = float(np.mean(left))
                    self.right_val = float(np.mean(right))
                    self.importance = max(total_var - best_mse, 0.0)

    def predict(self, X: NDArray) -> NDArray:
        """Predict using the split."""
        return np.where(
            X[:, self.feature_idx] <= self.threshold,
            self.left_val,
            self.right_val,
        )


def _train_random_forest(
    X: NDArray, y: NDArray,
    n_trees: int = 100, seed: int = 42,
) -> tuple[list[_DecisionStump], NDArray[np.float64]]:
    """
    Train a simplified Random Forest (ensemble of decision stumps).

    Returns:
        (trees, feature_importance) – importance is normalized, sum=1
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    trees: list[_DecisionStump] = []
    importance_sum = np.zeros(d)

    for t in range(n_trees):
        # Bootstrap sample
        idx = rng.choice(n, n, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]

        stump = _DecisionStump()
        stump.fit(X_boot, y_boot, rng)
        trees.append(stump)
        importance_sum[stump.feature_idx] += stump.importance

    total = np.sum(importance_sum)
    if total > 0:
        importance_sum /= total

    return trees, importance_sum


def _rf_predict(trees: list[_DecisionStump], X: NDArray) -> NDArray:
    """Predict by averaging all stumps."""
    preds = np.column_stack([t.predict(X) for t in trees])
    return np.mean(preds, axis=1)


# ──────────────────────────────────────────────────────────────────
# Neural Network (single hidden layer, for screening)
# ──────────────────────────────────────────────────────────────────

def _train_nn_screening(
    X: NDArray, y: NDArray,
    hidden: int = 32, epochs: int = 200,
    lr: float = 0.005, seed: int = 42,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Train a 1-hidden-layer NN. Returns (W1, b1, W2, b2)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape

    W1 = rng.normal(0, np.sqrt(2 / d), (d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, 1))
    b2 = np.zeros(1)

    for _ in range(epochs):
        # Forward
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)
        y_pred = (a1 @ W2 + b2).flatten()

        # Backward
        err = y_pred - y
        dW2 = a1.T @ err.reshape(-1, 1) / n
        db2 = np.mean(err)
        da1 = err.reshape(-1, 1) @ W2.T
        da1[z1 <= 0] = 0
        dW1 = X.T @ da1 / n
        db1 = np.mean(da1, axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    return W1, b1, W2, b2


def _nn_predict(
    X: NDArray, W1: NDArray, b1: NDArray, W2: NDArray, b2: NDArray,
) -> NDArray:
    """Forward pass prediction."""
    a1 = np.maximum(0, X @ W1 + b1)
    return (a1 @ W2 + b2).flatten()


# ──────────────────────────────────────────────────────────────────
# SVR (ε-SVR via SGD, simplified linear kernel)
# ──────────────────────────────────────────────────────────────────

def _train_svr(
    X: NDArray, y: NDArray,
    epsilon: float = 0.1, C: float = 1.0,
    epochs: int = 200, lr: float = 0.01, seed: int = 42,
) -> tuple[NDArray, float]:
    """
    Train linear SVR via SGD.

    ε-insensitive loss: L = max(0, |y - f(x)| - ε) + 0.5*C*||w||²

    Returns:
        (weights, bias)
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0, 0.01, d)
    b = 0.0

    for _ in range(epochs):
        idx = rng.permutation(n)
        for i in idx:
            pred = float(X[i] @ w + b)
            residual = y[i] - pred
            abs_r = abs(residual)

            if abs_r > epsilon:
                sign = 1.0 if residual > 0 else -1.0
                w += lr * (sign * X[i] - C * w / n)
                b += lr * sign
            else:
                w -= lr * C * w / n

    return w, b


def _svr_predict(X: NDArray, w: NDArray, b: float) -> NDArray:
    """SVR prediction."""
    return X @ w + b


# ──────────────────────────────────────────────────────────────────
# Permutation importance
# ──────────────────────────────────────────────────────────────────

def _permutation_importance(
    predict_fn, X: NDArray, y: NDArray,
    n_repeats: int = 5, seed: int = 42,
) -> NDArray[np.float64]:
    """
    Compute permutation feature importance.

    For each feature j:
        1. Compute baseline MSE
        2. Shuffle feature j, compute new MSE
        3. Importance = increase in MSE
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape

    baseline_pred = predict_fn(X)
    baseline_mse = float(np.mean((y - baseline_pred) ** 2))

    importance = np.zeros(d)

    for j in range(d):
        mse_increase = 0.0
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_pred = predict_fn(X_perm)
            perm_mse = float(np.mean((y - perm_pred) ** 2))
            mse_increase += (perm_mse - baseline_mse)
        importance[j] = mse_increase / n_repeats

    # Normalize
    importance = np.maximum(importance, 0.0)
    total = np.sum(importance)
    if total > 0:
        importance /= total

    return importance


# ──────────────────────────────────────────────────────────────────
# Borda count aggregation
# ──────────────────────────────────────────────────────────────────

def _borda_count(
    importances: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    Aggregate feature rankings via Borda count.

    Each method ranks features 1..d. Borda score = sum of ranks.
    Higher = more important.
    """
    d = len(importances[0])
    scores = np.zeros(d)

    for imp in importances:
        ranks = np.argsort(np.argsort(imp))  # Rank order
        scores += ranks

    return scores


# ──────────────────────────────────────────────────────────────────
# Main screening function
# ──────────────────────────────────────────────────────────────────

def screen_covariates(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    feature_names: list[str],
    top_k: int = 3,
    n_trees: int = 100,
    seed: int = 42,
) -> ScreeningResult:
    """
    Screen covariates using RF + NN + SVR with Borda count aggregation.

    Args:
        X:              (n, d) covariate matrix
        y:              (n,) PK parameter values
        feature_names:  Names of covariates
        top_k:          Number of covariates to select
        n_trees:        Number of RF trees
        seed:           Random seed

    Returns:
        ScreeningResult with rankings and selected covariates
    """
    n, d = X.shape

    # Standardize
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    x_std = np.where(x_std < 1e-10, 1.0, x_std)
    X_s = (X - x_mean) / x_std
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std < 1e-10:
        y_std = 1.0
    y_s = (y - y_mean) / y_std

    # 1. Random Forest
    trees, rf_imp = _train_random_forest(X_s, y_s, n_trees, seed)

    # 2. Neural Network
    W1, b1, W2, b2 = _train_nn_screening(X_s, y_s, seed=seed)
    nn_imp = _permutation_importance(
        lambda x: _nn_predict(x, W1, b1, W2, b2),
        X_s, y_s, seed=seed,
    )

    # 3. SVR
    w_svr, b_svr = _train_svr(X_s, y_s, seed=seed)
    svr_imp = _permutation_importance(
        lambda x: _svr_predict(x, w_svr, b_svr),
        X_s, y_s, seed=seed,
    )

    # Borda count
    borda = _borda_count([rf_imp, nn_imp, svr_imp])

    # Ranking
    rank_idx = np.argsort(-borda)  # Descending
    ranking = [feature_names[i] for i in rank_idx]
    selected = ranking[:top_k]

    return ScreeningResult(
        feature_names=feature_names,
        rf_importance=rf_imp,
        nn_importance=nn_imp,
        svr_importance=svr_imp,
        borda_scores=borda,
        ranking=ranking,
        selected=selected,
    )
