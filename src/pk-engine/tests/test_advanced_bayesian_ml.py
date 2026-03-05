"""
Tests for RF/NN/SVR screening, ADVI, EP, SMC, BMA – Phase 3+5 completion.

Tests:
    1. ML screening: RF, NN, SVR importance + Borda ranking
    2. ADVI: convergence, parameter estimation
    3. EP: convergence, covariance structure
    4. SMC: particle diversity, sequential processing
    5. BMA: model weights, combined parameters
"""

import numpy as np
import pytest

from pk.models import (
    DoseEvent, Gender, ModelType, Observation, PKParams, Route,
)
from pk.population import (
    VANCOMYCIN_VN, apply_iiv, compute_vancomycin_tv,
)
from pk.solver import predict_concentrations

from ai.ml_screening import screen_covariates, ScreeningResult
from bayesian.advi import run_advi, ADVIResult
from bayesian.ep import run_ep, EPResult
from bayesian.smc import run_smc, SMCResult
from bayesian.bma import run_bma, BMAResult


# ══════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════

def _make_pk_test_data() -> tuple[PKParams, list[DoseEvent], list[Observation]]:
    """Simulated vancomycin patient with noise-free TDM."""
    from pk.models import PatientData
    patient = PatientData(
        age=55, weight=75, height=170,
        gender=Gender.MALE, serum_creatinine=1.2,
    )
    tv = compute_vancomycin_tv(patient)
    true_eta = np.array([0.08, -0.04, 0.0, 0.0])
    true_params = apply_iiv(tv, true_eta)

    doses = [
        DoseEvent(time=0, amount=1000, duration=1.0, route=Route.IV_INFUSION),
        DoseEvent(time=12, amount=1000, duration=1.0, route=Route.IV_INFUSION),
    ]

    obs_times = [11.5, 23.5]
    c_true = predict_concentrations(
        true_params, doses, obs_times, ModelType.TWO_COMP_IV,
    )

    observations = [
        Observation(time=11.5, concentration=float(c_true[0]),
                    sample_type="trough"),
        Observation(time=23.5, concentration=float(c_true[1]),
                    sample_type="trough"),
    ]

    return tv, doses, observations


def _generate_screening_data(
    n: int = 80, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """CrCL, Weight, Age, Random -> CL."""
    rng = np.random.default_rng(seed)
    crcl = rng.uniform(30, 150, n)
    wt = rng.uniform(45, 120, n)
    age = rng.uniform(20, 85, n)
    random_noise_feat = rng.normal(0, 1, n)  # Irrelevant feature

    X = np.column_stack([crcl, wt, age, random_noise_feat])
    cl = 3.5 * (crcl / 85) ** 0.75 * (wt / 70) ** 0.75 + rng.normal(0, 0.2, n)
    return X, cl, ["CrCL", "Weight", "Age", "RandomNoise"]


# ══════════════════════════════════════════════════════════════════
# 1. ML Screening
# ══════════════════════════════════════════════════════════════════

class TestMLScreening:
    """Test RF/NN/SVR covariate screening."""

    def test_screening_runs(self) -> None:
        """Screening should complete without error."""
        X, y, names = _generate_screening_data()
        result = screen_covariates(X, y, names, top_k=2, seed=42)
        assert isinstance(result, ScreeningResult)
        assert len(result.selected) == 2

    def test_important_covariates_ranked_high(self) -> None:
        """CrCL and Weight should rank higher than RandomNoise."""
        X, y, names = _generate_screening_data(n=120)
        result = screen_covariates(X, y, names, top_k=2, seed=42)

        # RandomNoise should NOT be in top 2
        assert "RandomNoise" not in result.selected

    def test_rf_importance_sums_to_one(self) -> None:
        """RF importance should be normalized."""
        X, y, names = _generate_screening_data()
        result = screen_covariates(X, y, names, seed=42)
        assert abs(np.sum(result.rf_importance) - 1.0) < 0.01

    def test_ranking_length(self) -> None:
        """Ranking should include all features."""
        X, y, names = _generate_screening_data()
        result = screen_covariates(X, y, names, seed=42)
        assert len(result.ranking) == 4
        assert set(result.ranking) == set(names)


# ══════════════════════════════════════════════════════════════════
# 2. ADVI
# ══════════════════════════════════════════════════════════════════

class TestADVI:
    """Test ADVI posterior approximation."""

    def test_advi_runs(self) -> None:
        """ADVI should complete without error."""
        tv, doses, obs = _make_pk_test_data()
        result = run_advi(
            VANCOMYCIN_VN, tv, doses, obs, n_mc=5, max_iterations=50,
        )
        assert isinstance(result, ADVIResult)

    def test_advi_positive_params(self) -> None:
        """ADVI should produce positive PK parameters."""
        tv, doses, obs = _make_pk_test_data()
        result = run_advi(
            VANCOMYCIN_VN, tv, doses, obs, n_mc=5, max_iterations=50,
        )
        assert result.params.CL > 0
        assert result.params.V1 > 0

    def test_advi_positive_sigma(self) -> None:
        """Variational sigma should be positive."""
        tv, doses, obs = _make_pk_test_data()
        result = run_advi(
            VANCOMYCIN_VN, tv, doses, obs, n_mc=5, max_iterations=50,
        )
        assert np.all(result.sigma > 0)

    def test_advi_no_obs_raises(self) -> None:
        """ADVI should raise error with no observations."""
        tv, doses, _ = _make_pk_test_data()
        with pytest.raises(ValueError, match="observation"):
            run_advi(VANCOMYCIN_VN, tv, doses, [])


# ══════════════════════════════════════════════════════════════════
# 3. EP (Expectation Propagation)
# ══════════════════════════════════════════════════════════════════

class TestEP:
    """Test Expectation Propagation."""

    def test_ep_runs(self) -> None:
        """EP should complete without error."""
        tv, doses, obs = _make_pk_test_data()
        result = run_ep(VANCOMYCIN_VN, tv, doses, obs, max_iterations=5)
        assert isinstance(result, EPResult)

    def test_ep_positive_params(self) -> None:
        """EP should produce positive PK parameters."""
        tv, doses, obs = _make_pk_test_data()
        result = run_ep(VANCOMYCIN_VN, tv, doses, obs, max_iterations=5)
        assert result.params.CL > 0
        assert result.params.V1 > 0

    def test_ep_covariance_symmetric(self) -> None:
        """EP posterior covariance should be symmetric."""
        tv, doses, obs = _make_pk_test_data()
        result = run_ep(VANCOMYCIN_VN, tv, doses, obs, max_iterations=5)
        np.testing.assert_allclose(
            result.cov, result.cov.T, atol=1e-10,
        )

    def test_ep_no_obs_raises(self) -> None:
        """EP should raise error with no observations."""
        tv, doses, _ = _make_pk_test_data()
        with pytest.raises(ValueError, match="observation"):
            run_ep(VANCOMYCIN_VN, tv, doses, [])


# ══════════════════════════════════════════════════════════════════
# 4. SMC Particle Filter
# ══════════════════════════════════════════════════════════════════

class TestSMC:
    """Test SMC particle filter."""

    def test_smc_runs(self) -> None:
        """SMC should complete without error."""
        tv, doses, obs = _make_pk_test_data()
        result = run_smc(
            VANCOMYCIN_VN, tv, doses, obs,
            n_particles=50, seed=42,
        )
        assert isinstance(result, SMCResult)

    def test_smc_particle_count(self) -> None:
        """Should have correct number of particles."""
        tv, doses, obs = _make_pk_test_data()
        result = run_smc(
            VANCOMYCIN_VN, tv, doses, obs,
            n_particles=100, seed=42,
        )
        assert result.particles.shape[0] == 100
        assert result.particles.shape[1] == 4  # CL, V1, Q, V2

    def test_smc_weights_sum_to_one(self) -> None:
        """Particle weights should sum to 1."""
        tv, doses, obs = _make_pk_test_data()
        result = run_smc(
            VANCOMYCIN_VN, tv, doses, obs,
            n_particles=50, seed=42,
        )
        assert abs(np.sum(result.weights) - 1.0) < 1e-10

    def test_smc_ess_history_length(self) -> None:
        """ESS history should have one entry per observation."""
        tv, doses, obs = _make_pk_test_data()
        result = run_smc(
            VANCOMYCIN_VN, tv, doses, obs,
            n_particles=50, seed=42,
        )
        assert len(result.ess_history) == len(obs)

    def test_smc_positive_params(self) -> None:
        """SMC should produce positive PK parameters."""
        tv, doses, obs = _make_pk_test_data()
        result = run_smc(
            VANCOMYCIN_VN, tv, doses, obs,
            n_particles=100, seed=42,
        )
        assert result.params.CL > 0
        assert result.params.V1 > 0


# ══════════════════════════════════════════════════════════════════
# 5. BMA (Bayesian Model Averaging)
# ══════════════════════════════════════════════════════════════════

class TestBMA:
    """Test Bayesian Model Averaging."""

    def test_bma_runs(self) -> None:
        """BMA should complete without error."""
        tv, doses, obs = _make_pk_test_data()
        # Use same model twice with slightly different TV
        tv2 = PKParams(CL=tv.CL * 1.1, V1=tv.V1, Q=tv.Q, V2=tv.V2)
        result = run_bma(
            [VANCOMYCIN_VN, VANCOMYCIN_VN],
            [tv, tv2], doses, obs,
        )
        assert isinstance(result, BMAResult)

    def test_bma_weights_sum_to_one(self) -> None:
        """Model weights should sum to 1."""
        tv, doses, obs = _make_pk_test_data()
        tv2 = PKParams(CL=tv.CL * 1.2, V1=tv.V1 * 0.9, Q=tv.Q, V2=tv.V2)
        result = run_bma(
            [VANCOMYCIN_VN, VANCOMYCIN_VN],
            [tv, tv2], doses, obs,
        )
        assert abs(np.sum(result.model_weights) - 1.0) < 1e-10

    def test_bma_combined_params_positive(self) -> None:
        """Combined parameters should be positive."""
        tv, doses, obs = _make_pk_test_data()
        tv2 = PKParams(CL=tv.CL * 1.1, V1=tv.V1, Q=tv.Q, V2=tv.V2)
        result = run_bma(
            [VANCOMYCIN_VN, VANCOMYCIN_VN],
            [tv, tv2], doses, obs,
        )
        assert result.combined_params.CL > 0
        assert result.combined_params.V1 > 0

    def test_bma_too_few_models_raises(self) -> None:
        """BMA should raise error with fewer than 2 models."""
        tv, doses, obs = _make_pk_test_data()
        with pytest.raises(ValueError, match="2 models"):
            run_bma(
                [VANCOMYCIN_VN], [tv], doses, obs,
            )

    def test_bma_stacking_method(self) -> None:
        """Stacking method should also work."""
        tv, doses, obs = _make_pk_test_data()
        tv2 = PKParams(CL=tv.CL * 1.1, V1=tv.V1, Q=tv.Q, V2=tv.V2)
        result = run_bma(
            [VANCOMYCIN_VN, VANCOMYCIN_VN],
            [tv, tv2], doses, obs, method="stacking",
        )
        assert result.method == "stacking"
        assert abs(np.sum(result.model_weights) - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
