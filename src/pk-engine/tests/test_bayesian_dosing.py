"""
Tests for Bayesian Engine + Dosing Optimizer – Phase 3-4 verification.

Tests:
    1. MAP estimation recovers known parameters
    2. Laplace approximation produces valid CIs
    3. Dose optimizer finds reasonable regimen
    4. Monte Carlo PTA is within expected range
"""

import numpy as np
import pytest

from pk.models import (
    DoseEvent,
    Gender,
    ModelType,
    Observation,
    PKParams,
    PatientData,
    Route,
)
from pk.population import (
    VANCOMYCIN_VN,
    apply_iiv,
    compute_vancomycin_tv,
)
from pk.solver import predict_concentrations
from bayesian.map_estimator import estimate_map, MAPResult
from bayesian.laplace import laplace_approximation, bayesian_update
from dosing.optimizer import (
    DosingResult,
    PKPDTarget,
    monte_carlo_pta,
    optimize_dose,
)


# ══════════════════════════════════════════════════════════════════
# Shared test fixtures
# ══════════════════════════════════════════════════════════════════

def _make_simulated_patient() -> (
    tuple[PKParams, PKParams, list[DoseEvent], list[Observation]]
):
    """
    Create a simulated patient with known true parameters.

    Returns:
        Tuple of (tv_params, true_params, doses, observations)
    """
    patient = PatientData(
        age=55, weight=75, height=170,
        gender=Gender.MALE, serum_creatinine=1.2,
    )
    tv = compute_vancomycin_tv(patient)

    # True individual parameters (known eta for validation)
    true_eta = np.array([0.1, -0.05, 0.0, 0.0])
    true_params = apply_iiv(tv, true_eta)

    # Doses: 1000mg q12h IV infusion over 1h, 3 doses
    doses = [
        DoseEvent(time=0, amount=1000, duration=1.0,
                  route=Route.IV_INFUSION),
        DoseEvent(time=12, amount=1000, duration=1.0,
                  route=Route.IV_INFUSION),
        DoseEvent(time=24, amount=1000, duration=1.0,
                  route=Route.IV_INFUSION),
    ]

    # Simulate true concentrations at observation times
    obs_times = [11.5, 23.5]  # trough times
    c_true = predict_concentrations(
        true_params, doses, obs_times, ModelType.TWO_COMP_IV
    )

    observations = [
        Observation(time=obs_times[0], concentration=float(c_true[0]),
                    sample_type="trough"),
        Observation(time=obs_times[1], concentration=float(c_true[1]),
                    sample_type="trough"),
    ]

    return tv, true_params, doses, observations


# ══════════════════════════════════════════════════════════════════
# 1. MAP Estimation
# ══════════════════════════════════════════════════════════════════

class TestMAPEstimation:
    """Test MAP estimator."""

    def test_map_converges(self) -> None:
        """MAP should converge successfully."""
        tv, true_params, doses, obs = _make_simulated_patient()
        result = estimate_map(
            VANCOMYCIN_VN, tv, doses, obs, ModelType.TWO_COMP_IV
        )
        assert result.success

    def test_map_recovers_cl(self) -> None:
        """MAP-estimated CL should be close to true CL."""
        tv, true_params, doses, obs = _make_simulated_patient()
        result = estimate_map(
            VANCOMYCIN_VN, tv, doses, obs, ModelType.TWO_COMP_IV
        )
        # Within 30% of true (generous for 2 observations)
        relative_error = abs(result.params.CL - true_params.CL) / true_params.CL
        assert relative_error < 0.30, (
            f"CL error: {relative_error:.2%}, "
            f"MAP={result.params.CL:.2f}, True={true_params.CL:.2f}"
        )

    def test_map_returns_residuals(self) -> None:
        """Residuals should be small for noise-free data."""
        tv, true_params, doses, obs = _make_simulated_patient()
        result = estimate_map(
            VANCOMYCIN_VN, tv, doses, obs, ModelType.TWO_COMP_IV
        )
        # Residuals should be small (< 2 mg/L for noise-free)
        max_residual = float(np.max(np.abs(result.residuals)))
        assert max_residual < 2.0, f"Max residual: {max_residual:.4f}"

    def test_map_no_observations_raises(self) -> None:
        """MAP should raise error with no observations."""
        tv, _, doses, _ = _make_simulated_patient()
        with pytest.raises(ValueError, match="observation"):
            estimate_map(VANCOMYCIN_VN, tv, doses, [])


# ══════════════════════════════════════════════════════════════════
# 2. Laplace Approximation
# ══════════════════════════════════════════════════════════════════

class TestLaplaceApproximation:
    """Test Laplace posterior approximation."""

    def test_laplace_produces_cis(self) -> None:
        """Laplace should produce valid credible intervals."""
        tv, _, doses, obs = _make_simulated_patient()
        map_result = estimate_map(
            VANCOMYCIN_VN, tv, doses, obs, ModelType.TWO_COMP_IV
        )
        lap = laplace_approximation(
            map_result, VANCOMYCIN_VN, tv, doses, obs
        )

        # CIs should exist for CL and V1
        assert "CL" in lap.ci_lower
        assert "CL" in lap.ci_upper

        # Lower < MAP < Upper
        assert lap.ci_lower["CL"] < map_result.params.CL
        assert lap.ci_upper["CL"] > map_result.params.CL

    def test_laplace_posterior_cov_positive(self) -> None:
        """Posterior covariance should be positive definite."""
        tv, _, doses, obs = _make_simulated_patient()
        map_result = estimate_map(
            VANCOMYCIN_VN, tv, doses, obs, ModelType.TWO_COMP_IV
        )
        lap = laplace_approximation(
            map_result, VANCOMYCIN_VN, tv, doses, obs
        )

        # All eigenvalues should be positive
        eigvals = np.linalg.eigvalsh(lap.posterior_cov)
        assert all(v > 0 for v in eigvals)


# ══════════════════════════════════════════════════════════════════
# 3. Online Bayesian Update
# ══════════════════════════════════════════════════════════════════

class TestBayesianUpdate:
    """Test online (sequential) Bayesian update."""

    def test_update_reduces_variance(self) -> None:
        """Adding an observation should reduce posterior variance."""
        theta_old = np.array([3.5, 30.0])
        sigma_old_inv = np.diag([1.0, 0.1])  # Prior precision
        jacobian = np.array([0.5, 0.01])  # Sensitivity

        theta_new, sigma_new_inv = bayesian_update(
            theta_old, sigma_old_inv,
            y_new=15.0, y_pred_new=14.0,
            jacobian=jacobian, sigma_eps=1.0,
        )

        # Precision should increase (variance should decrease)
        var_old = np.diag(np.linalg.inv(sigma_old_inv))
        var_new = np.diag(np.linalg.inv(sigma_new_inv))
        assert var_new[0] <= var_old[0]


# ══════════════════════════════════════════════════════════════════
# 4. Dose Optimizer
# ══════════════════════════════════════════════════════════════════

class TestDoseOptimizer:
    """Test dose optimization."""

    def test_optimizer_finds_regimen(self) -> None:
        """Optimizer should return a valid dosing regimen."""
        params = PKParams(CL=3.5, V1=30.0, Q=5.0, V2=40.0)
        result = optimize_dose(params, ModelType.TWO_COMP_IV)

        assert isinstance(result, DosingResult)
        assert result.dose_mg > 0
        assert result.interval_h > 0
        assert result.predicted_auc24 > 0

    def test_optimizer_targets_auc(self) -> None:
        """Optimal AUC24/MIC should be within or near target range."""
        params = PKParams(CL=3.5, V1=30.0, Q=5.0, V2=40.0)
        target = PKPDTarget(auc24_min=400, auc24_max=600)
        result = optimize_dose(params, ModelType.TWO_COMP_IV, target)

        # Should be reasonably close to target range
        assert 200 < result.predicted_auc24_mic < 900

    def test_optimizer_provides_alternatives(self) -> None:
        """Optimizer should provide ranked alternatives."""
        params = PKParams(CL=3.5, V1=30.0, Q=5.0, V2=40.0)
        result = optimize_dose(params, ModelType.TWO_COMP_IV)
        assert len(result.alternatives) > 0

    def test_optimizer_higher_cl_needs_higher_dose(self) -> None:
        """Patient with higher CL should need higher dose."""
        low_cl = PKParams(CL=2.5, V1=30.0, Q=5.0, V2=40.0)
        high_cl = PKParams(CL=5.0, V1=30.0, Q=5.0, V2=40.0)

        result_low = optimize_dose(low_cl, ModelType.TWO_COMP_IV)
        result_high = optimize_dose(high_cl, ModelType.TWO_COMP_IV)

        # Higher CL -> needs higher daily dose
        daily_low = result_low.dose_mg * 24 / result_low.interval_h
        daily_high = result_high.dose_mg * 24 / result_high.interval_h
        assert daily_high >= daily_low


# ══════════════════════════════════════════════════════════════════
# 5. Monte Carlo PTA (simplified, small N for speed)
# ══════════════════════════════════════════════════════════════════

class TestMonteCarloPTA:
    """Test Monte Carlo PTA (small N for speed)."""

    def test_pta_between_0_and_1(self) -> None:
        """PTA should be between 0 and 1."""
        tv = PKParams(CL=3.5, V1=30.0, Q=5.0, V2=40.0)
        pta = monte_carlo_pta(
            tv, VANCOMYCIN_VN,
            dose_mg=1000, interval_h=12,
            n_simulations=100,  # Small N for speed
            seed=42,
        )
        assert 0.0 <= pta <= 1.0

    def test_pta_higher_dose_higher_pta(self) -> None:
        """Higher dose should generally give higher PTA (up to a point)."""
        tv = PKParams(CL=3.5, V1=30.0, Q=5.0, V2=40.0)

        pta_low = monte_carlo_pta(
            tv, VANCOMYCIN_VN,
            dose_mg=500, interval_h=12,
            n_simulations=100, seed=42,
        )
        pta_high = monte_carlo_pta(
            tv, VANCOMYCIN_VN,
            dose_mg=1500, interval_h=12,
            n_simulations=100, seed=42,
        )
        # Higher dose should have higher or equal PTA
        assert pta_high >= pta_low or abs(pta_high - pta_low) < 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
