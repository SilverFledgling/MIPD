"""
Tests for GP Covariate Model and BNN/MC-Dropout – Phase 5 AI/ML.

Tests:
    1. GP: Training, prediction, uncertainty, kernel correctness
    2. BNN: Training, MC Dropout prediction, uncertainty quantification
"""

import numpy as np
import pytest

from ai.gp_covariate import (
    GPCovariateModel,
    GPPrediction,
    rbf_kernel,
    train_gp,
    predict_gp,
)
from ai.bnn_model import (
    BNNModel,
    BNNPrediction,
    train_bnn,
    predict_bnn,
)


# ══════════════════════════════════════════════════════════════════
# Helper: generate synthetic covariate-PK data
# ══════════════════════════════════════════════════════════════════

def _generate_synthetic_data(
    n: int = 100, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate CrCL + Weight -> CL relationship.
    CL = 3.5 * (CrCL/85)^0.75 * (WT/70)^0.75 + noise
    """
    rng = np.random.default_rng(seed)
    crcl = rng.uniform(30, 150, n)
    wt = rng.uniform(45, 120, n)
    X = np.column_stack([crcl, wt])

    cl = 3.5 * (crcl / 85) ** 0.75 * (wt / 70) ** 0.75
    noise = rng.normal(0, 0.3, n)
    y = cl + noise

    return X, y, ["CrCL", "Weight"]


# ══════════════════════════════════════════════════════════════════
# 1. RBF Kernel
# ══════════════════════════════════════════════════════════════════

class TestRBFKernel:
    """Test RBF kernel properties."""

    def test_kernel_symmetric(self) -> None:
        """K(X, X) should be symmetric."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        l = np.array([1.0, 1.0])
        K = rbf_kernel(X, X, l, 1.0)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_kernel_diagonal_equals_signal_var(self) -> None:
        """k(x, x) = sigma_f^2."""
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        l = np.array([1.0, 1.0])
        sigma_f2 = 2.5
        K = rbf_kernel(X, X, l, sigma_f2)
        np.testing.assert_allclose(np.diag(K), sigma_f2, atol=1e-12)

    def test_kernel_positive_definite(self) -> None:
        """Kernel matrix should be positive semi-definite."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (10, 3))
        K = rbf_kernel(X, X, np.ones(3), 1.0)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-10)


# ══════════════════════════════════════════════════════════════════
# 2. GP Training & Prediction
# ══════════════════════════════════════════════════════════════════

class TestGPCovariate:
    """Test GP covariate model."""

    def test_gp_trains(self) -> None:
        """GP should train without error."""
        X, y, names = _generate_synthetic_data(50)
        model = train_gp(X, y, "CL", names)
        assert isinstance(model, GPCovariateModel)
        assert model.parameter_name == "CL"

    def test_gp_predicts_training_data(self) -> None:
        """GP should fit training data well (low error)."""
        X, y, names = _generate_synthetic_data(50)
        model = train_gp(X, y, "CL", names)
        preds = predict_gp(model, X)

        errors = np.array([p.mean - y[i] for i, p in enumerate(preds)])
        rmse = np.sqrt(np.mean(errors ** 2))
        # Should be small since GP interpolates training data
        assert rmse < 1.0, f"RMSE = {rmse:.3f}"

    def test_gp_provides_uncertainty(self) -> None:
        """GP should provide positive uncertainty."""
        X, y, names = _generate_synthetic_data(50)
        model = train_gp(X, y, "CL", names)

        # Predict at a point far from training data
        X_far = np.array([[200.0, 200.0]])
        preds = predict_gp(model, X_far)
        assert preds[0].std > 0
        assert preds[0].ci95_lower < preds[0].ci95_upper

    def test_gp_uncertainty_increases_far_from_data(self) -> None:
        """Uncertainty should be larger far from training data."""
        X, y, names = _generate_synthetic_data(50)
        model = train_gp(X, y, "CL", names)

        # Near training data
        X_near = np.array([[85.0, 70.0]])
        pred_near = predict_gp(model, X_near)

        # Far from training data
        X_far = np.array([[300.0, 200.0]])
        pred_far = predict_gp(model, X_far)

        assert pred_far[0].std > pred_near[0].std

    def test_gp_batch_prediction(self) -> None:
        """GP should handle batch predictions."""
        X, y, names = _generate_synthetic_data(50)
        model = train_gp(X, y, "CL", names)

        X_test = np.array([[80, 70], [60, 90], [100, 50]])
        preds = predict_gp(model, X_test)
        assert len(preds) == 3
        for p in preds:
            assert isinstance(p, GPPrediction)
            assert p.mean > 0  # CL should be positive


# ══════════════════════════════════════════════════════════════════
# 3. BNN Training & Prediction
# ══════════════════════════════════════════════════════════════════

class TestBNN:
    """Test BNN with MC Dropout."""

    def test_bnn_trains(self) -> None:
        """BNN should train without error."""
        X, y, names = _generate_synthetic_data(100)
        model = train_bnn(
            X, y, "CL", names,
            hidden_sizes=(32, 16), n_epochs=100, seed=42,
        )
        assert isinstance(model, BNNModel)
        assert model.parameter_name == "CL"
        assert len(model.weights) == 3  # 2 hidden + 1 output

    def test_bnn_predicts_reasonable(self) -> None:
        """BNN predictions should be in reasonable range."""
        X, y, names = _generate_synthetic_data(100)
        model = train_bnn(
            X, y, "CL", names,
            hidden_sizes=(64, 32), n_epochs=300, seed=42,
        )

        X_test = np.array([[85, 70], [50, 80]])
        preds = predict_bnn(model, X_test, n_mc_samples=50, seed=42)

        for p in preds:
            assert isinstance(p, BNNPrediction)
            # CL should be in reasonable pharmacological range
            assert 0.5 < p.mean < 10.0, f"CL = {p.mean:.2f}"

    def test_bnn_provides_uncertainty(self) -> None:
        """BNN should provide positive uncertainty via MC Dropout."""
        X, y, names = _generate_synthetic_data(100)
        model = train_bnn(
            X, y, "CL", names,
            hidden_sizes=(32, 16), n_epochs=200,
            dropout_rate=0.3, seed=42,
        )

        preds = predict_bnn(
            model, np.array([[85, 70]]),
            n_mc_samples=100, seed=42,
        )
        assert preds[0].std > 0
        assert preds[0].ci95_lower < preds[0].mean < preds[0].ci95_upper

    def test_bnn_mc_samples_count(self) -> None:
        """MC samples should match requested count."""
        X, y, names = _generate_synthetic_data(50)
        model = train_bnn(
            X, y, "CL", names,
            hidden_sizes=(16,), n_epochs=50, seed=42,
        )

        preds = predict_bnn(
            model, np.array([[85, 70]]),
            n_mc_samples=200, seed=42,
        )
        assert len(preds[0].mc_samples) == 200

    def test_bnn_different_seeds_give_different_mc(self) -> None:
        """Different seeds should give different MC samples."""
        X, y, names = _generate_synthetic_data(50)
        model = train_bnn(
            X, y, "CL", names,
            hidden_sizes=(16,), n_epochs=50, seed=42,
        )

        X_test = np.array([[85, 70]])
        pred1 = predict_bnn(model, X_test, n_mc_samples=50, seed=1)
        pred2 = predict_bnn(model, X_test, n_mc_samples=50, seed=99)

        # Means should differ slightly (different dropout masks)
        assert pred1[0].mean != pred2[0].mean


# ══════════════════════════════════════════════════════════════════
# 4. GP vs BNN comparison
# ══════════════════════════════════════════════════════════════════

class TestGPvsBNN:
    """Compare GP and BNN on same data."""

    def test_both_predict_positive_cl(self) -> None:
        """Both models should predict positive CL for normal patient."""
        X, y, names = _generate_synthetic_data(80)
        X_test = np.array([[85, 70]])

        gp = train_gp(X, y, "CL", names)
        bnn = train_bnn(
            X, y, "CL", names,
            hidden_sizes=(32, 16), n_epochs=200, seed=42,
        )

        gp_pred = predict_gp(gp, X_test)[0]
        bnn_pred = predict_bnn(bnn, X_test, n_mc_samples=50)[0]

        assert gp_pred.mean > 0
        assert bnn_pred.mean > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
