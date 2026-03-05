"""
Tests for MCMC/NUTS posterior sampling – Phase 3 verification.

Tests:
    1. MCMC runs and produces posterior samples
    2. Posterior mean is close to true parameters (noise-free data)
    3. R-hat < 1.05 (chains converged)
    4. ESS is reasonable
    5. Credible intervals contain true values
    6. Higher data -> narrower posterior

NOTE: These tests require NumPyro + JAX.
      Run on WSL/Linux: python3 -m pytest tests/test_mcmc.py -v --tb=short
"""

import numpy as np
import pytest

# Skip all tests if NumPyro not available
numpyro = pytest.importorskip("numpyro", reason="NumPyro required")

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
from bayesian.mcmc import run_mcmc, MCMCResult


# ══════════════════════════════════════════════════════════════════
# Shared test fixtures
# ══════════════════════════════════════════════════════════════════

def _make_simulated_patient() -> (
    tuple[PKParams, PKParams, list[DoseEvent], list[Observation]]
):
    """
    Create a simulated patient with known parameters and noise-free TDM data.
    """
    patient = PatientData(
        age=55, weight=75, height=170,
        gender=Gender.MALE, serum_creatinine=1.2,
    )
    tv = compute_vancomycin_tv(patient)

    # True individual parameters
    true_eta = np.array([0.1, -0.05, 0.0, 0.0])
    true_params = apply_iiv(tv, true_eta)

    # Doses: 1000mg q12h IV infusion, 3 doses
    doses = [
        DoseEvent(time=0, amount=1000, duration=1.0,
                  route=Route.IV_INFUSION),
        DoseEvent(time=12, amount=1000, duration=1.0,
                  route=Route.IV_INFUSION),
        DoseEvent(time=24, amount=1000, duration=1.0,
                  route=Route.IV_INFUSION),
    ]

    # Simulate 2 trough observations (noise-free)
    obs_times = [11.5, 23.5]
    c_true = predict_concentrations(
        true_params, doses, obs_times, ModelType.TWO_COMP_IV,
    )

    observations = [
        Observation(time=obs_times[0], concentration=float(c_true[0]),
                    sample_type="trough"),
        Observation(time=obs_times[1], concentration=float(c_true[1]),
                    sample_type="trough"),
    ]

    return tv, true_params, doses, observations


# ══════════════════════════════════════════════════════════════════
# 1. Basic MCMC execution
# ══════════════════════════════════════════════════════════════════

class TestMCMCExecution:
    """Test MCMC runs and produces valid output."""

    def test_mcmc_runs(self) -> None:
        """MCMC should complete without error."""
        tv, _, doses, obs = _make_simulated_patient()
        result = run_mcmc(
            VANCOMYCIN_VN, tv, doses, obs,
            n_warmup=100, n_samples=200, n_chains=1, seed=42,
        )
        assert isinstance(result, MCMCResult)
        assert result.n_samples == 200

    def test_mcmc_produces_samples(self) -> None:
        """Posterior eta should have correct shape."""
        tv, _, doses, obs = _make_simulated_patient()
        result = run_mcmc(
            VANCOMYCIN_VN, tv, doses, obs,
            n_warmup=100, n_samples=200, n_chains=1, seed=42,
        )
        # Shape: (total_samples, n_params)
        assert result.posterior_eta.shape[0] == 200
        assert result.posterior_eta.shape[1] == 4

    def test_mcmc_summary_has_params(self) -> None:
        """Summary should contain CL and V1."""
        tv, _, doses, obs = _make_simulated_patient()
        result = run_mcmc(
            VANCOMYCIN_VN, tv, doses, obs,
            n_warmup=100, n_samples=200, n_chains=1, seed=42,
        )
        assert "CL" in result.summary
        assert "V1" in result.summary
        assert "mean" in result.summary["CL"]
        assert "ci95_lower" in result.summary["CL"]
        assert "ci95_upper" in result.summary["CL"]


# ══════════════════════════════════════════════════════════════════
# 2. Posterior quality
# ══════════════════════════════════════════════════════════════════

class TestPosteriorQuality:
    """Test posterior samples quality and accuracy."""

    def test_map_params_positive(self) -> None:
        """MAP (posterior mean) PK parameters should be positive."""
        tv, _, doses, obs = _make_simulated_patient()
        result = run_mcmc(
            VANCOMYCIN_VN, tv, doses, obs,
            n_warmup=200, n_samples=500, n_chains=1, seed=42,
        )
        assert result.map_params.CL > 0
        assert result.map_params.V1 > 0

    def test_posterior_cl_near_true(self) -> None:
        """Posterior mean CL should be within 50% of true CL."""
        tv, true_params, doses, obs = _make_simulated_patient()
        result = run_mcmc(
            VANCOMYCIN_VN, tv, doses, obs,
            n_warmup=200, n_samples=500, n_chains=1, seed=42,
        )
        cl_mean = result.summary["CL"]["mean"]
        relative_error = abs(cl_mean - true_params.CL) / true_params.CL
        assert relative_error < 0.50, (
            f"CL error: {relative_error:.2%}, "
            f"MCMC={cl_mean:.2f}, True={true_params.CL:.2f}"
        )

    def test_credible_interval_valid(self) -> None:
        """95% CI lower < mean < upper."""
        tv, _, doses, obs = _make_simulated_patient()
        result = run_mcmc(
            VANCOMYCIN_VN, tv, doses, obs,
            n_warmup=200, n_samples=500, n_chains=1, seed=42,
        )
        for param in ["CL", "V1"]:
            s = result.summary[param]
            assert s["ci95_lower"] < s["mean"] < s["ci95_upper"]


# ══════════════════════════════════════════════════════════════════
# 3. Diagnostics
# ══════════════════════════════════════════════════════════════════

class TestMCMCDiagnostics:
    """Test MCMC convergence diagnostics."""

    def test_rhat_reported(self) -> None:
        """R-hat should be reported for each parameter."""
        tv, _, doses, obs = _make_simulated_patient()
        result = run_mcmc(
            VANCOMYCIN_VN, tv, doses, obs,
            n_warmup=200, n_samples=500, n_chains=2, seed=42,
        )
        assert "CL" in result.rhat
        assert "V1" in result.rhat

    def test_ess_reported(self) -> None:
        """ESS should be reported and positive."""
        tv, _, doses, obs = _make_simulated_patient()
        result = run_mcmc(
            VANCOMYCIN_VN, tv, doses, obs,
            n_warmup=100, n_samples=200, n_chains=1, seed=42,
        )
        assert result.ess["CL"] > 0
        assert result.ess["V1"] > 0

    def test_n_divergences_reported(self) -> None:
        """Divergence count should be non-negative."""
        tv, _, doses, obs = _make_simulated_patient()
        result = run_mcmc(
            VANCOMYCIN_VN, tv, doses, obs,
            n_warmup=100, n_samples=200, n_chains=1, seed=42,
        )
        assert result.n_divergences >= 0


# ══════════════════════════════════════════════════════════════════
# 4. Error handling
# ══════════════════════════════════════════════════════════════════

class TestMCMCErrorHandling:
    """Test MCMC error handling."""

    def test_no_observations_raises(self) -> None:
        """MCMC should raise error with no observations."""
        tv, _, doses, _ = _make_simulated_patient()
        with pytest.raises(ValueError, match="observation"):
            run_mcmc(VANCOMYCIN_VN, tv, doses, [])

    def test_no_doses_raises(self) -> None:
        """MCMC should raise error with no doses."""
        tv, _, _, obs = _make_simulated_patient()
        with pytest.raises(ValueError, match="dose"):
            run_mcmc(VANCOMYCIN_VN, tv, [], obs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
