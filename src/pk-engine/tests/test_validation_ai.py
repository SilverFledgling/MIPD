"""
Tests for Phase 5: Validation Metrics, Swift Hydra, SHAP, VPC.

Tests:
    1. Validation metrics (CCC, MPE/MAPE, RMSE, TOST, NPDE, Coverage)
    2. Swift Hydra anomaly detection (4 heads)
    3. SHAP explainer (covariate contributions)
    4. VPC simulation (percentile bands)
"""

import numpy as np
import pytest

from pk.models import Gender, PatientData, PKParams, DoseEvent, Route, ModelType
from pk.population import VANCOMYCIN_VN
from validation.metrics import (
    compute_metrics,
    concordance_correlation,
    tost_equivalence,
    compute_npde,
    coverage_probability,
)
from ai.anomaly_detection import detect_anomaly, QualityVerdict
from ai.shap_explainer import explain_pk_parameter
from validation.vpc import simulate_vpc


# ══════════════════════════════════════════════════════════════════
# 1. Validation Metrics
# ══════════════════════════════════════════════════════════════════

class TestValidationMetrics:
    """Test prediction accuracy metrics."""

    def test_perfect_prediction(self) -> None:
        """Perfect prediction -> CCC=1, PE=0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = compute_metrics(x, x)
        assert abs(m.ccc - 1.0) < 0.001
        assert abs(m.mpe) < 0.001
        assert abs(m.rmse) < 0.001

    def test_biased_prediction(self) -> None:
        """Constant 10% overestimation -> MPE ~10%."""
        true_vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        estimated = true_vals * 1.10
        m = compute_metrics(estimated, true_vals)
        assert abs(m.mpe - 10.0) < 1.0
        assert m.ccc > 0.9  # Still correlated

    def test_ccc_uncorrelated(self) -> None:
        """Random data -> CCC near 0."""
        rng = np.random.default_rng(42)
        x = rng.normal(10, 2, 100)
        y = rng.normal(10, 2, 100)
        ccc = concordance_correlation(x, y)
        assert abs(ccc) < 0.3

    def test_bland_altman_limits(self) -> None:
        """LoA should contain ~95% of differences."""
        rng = np.random.default_rng(42)
        true_vals = rng.uniform(5, 40, 200)
        noise = rng.normal(0, 1, 200)
        estimated = true_vals + noise
        m = compute_metrics(estimated, true_vals)
        # LoA should be approximately ±2
        assert m.loa_upper > 0
        assert m.loa_lower < 0


class TestTOST:
    """Test TOST equivalence."""

    def test_equivalent_data(self) -> None:
        """Small bias within margin -> equivalent."""
        rng = np.random.default_rng(42)
        true_vals = rng.uniform(10, 30, 100)
        noise_pct = rng.normal(2, 3, 100)  # Mean 2% bias, SD 3%
        estimated = true_vals * (1 + noise_pct / 100)
        result = tost_equivalence(estimated, true_vals, margin=0.10)
        assert result.is_equivalent

    def test_non_equivalent_data(self) -> None:
        """Large bias outside margin -> not equivalent."""
        rng = np.random.default_rng(42)
        true_vals = rng.uniform(10, 30, 100)
        noise_pct = rng.normal(50, 5, 100)  # Mean 50% bias, SD 5%
        estimated = true_vals * (1 + noise_pct / 100)
        result = tost_equivalence(estimated, true_vals, margin=0.10)
        assert not result.is_equivalent

    def test_ci90_contains_mean(self) -> None:
        """90% CI should contain the mean PE."""
        true_vals = np.array([10.0, 20.0, 30.0] * 20)
        estimated = true_vals * 1.05
        result = tost_equivalence(estimated, true_vals, margin=0.10)
        assert result.ci90_lower <= result.mean_diff <= result.ci90_upper


class TestNPDE:
    """Test NPDE computation."""

    def test_npde_well_specified(self) -> None:
        """Well-specified model -> NPDE ~ N(0,1)."""
        rng = np.random.default_rng(42)
        n = 50
        observed = rng.normal(15, 3, n)

        # Simulated matrix: observations from same distribution
        k = 1000
        sim_matrix = rng.normal(15, 3, (n, k))

        result = compute_npde(observed, sim_matrix)
        # Mean should be ~0
        assert abs(result.mean) < 0.5
        # Variance should be ~1
        assert 0.5 < result.variance < 2.0

    def test_npde_misspecified(self) -> None:
        """Misspecified model -> NPDE mean ≠ 0."""
        n = 50
        rng = np.random.default_rng(42)
        observed = rng.normal(20, 2, n)  # True distribution

        # Simulated from a different distribution (biased model)
        k = 1000
        sim_matrix = rng.normal(10, 2, (n, k))  # Model predicts lower

        result = compute_npde(observed, sim_matrix)
        # Mean should be significantly > 0 (observations above simulations)
        assert result.mean > 1.0


class TestCoverage:
    """Test coverage probability."""

    def test_perfect_coverage(self) -> None:
        """Wide CI -> 100% coverage."""
        true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        assert coverage_probability(true, lower, upper) == 1.0

    def test_zero_coverage(self) -> None:
        """CI entirely wrong -> 0% coverage."""
        true = np.array([10.0, 20.0, 30.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([5.0, 5.0, 5.0])
        assert coverage_probability(true, lower, upper) == 0.0


# ══════════════════════════════════════════════════════════════════
# 2. Swift Hydra Anomaly Detection
# ══════════════════════════════════════════════════════════════════

class TestSwiftHydra:
    """Test 4-head anomaly detection."""

    def test_normal_sample_accepted(self) -> None:
        """Normal TDM sample -> ACCEPT."""
        result = detect_anomaly(
            c_obs=15.0,
            c_predicted=14.5,
            population_mean=15.0,
            population_sd=5.0,
            residual_sd=2.0,
            omega_cl=0.3,
        )
        assert result.verdict == QualityVerdict.ACCEPT
        assert result.quality_score >= 0.8
        assert result.is_valid

    def test_extreme_sample_rejected(self) -> None:
        """Extreme outlier -> REJECT."""
        result = detect_anomaly(
            c_obs=100.0,  # Way too high
            c_predicted=15.0,
            population_mean=15.0,
            population_sd=5.0,
            residual_sd=2.0,
            omega_cl=0.3,
        )
        assert result.verdict == QualityVerdict.REJECT
        assert result.quality_score < 0.5
        assert not result.is_valid

    def test_borderline_sample_warning(self) -> None:
        """Borderline sample -> WARNING."""
        result = detect_anomaly(
            c_obs=28.0,  # High but not extreme
            c_predicted=15.0,
            population_mean=15.0,
            population_sd=5.0,
            residual_sd=2.0,
            omega_cl=0.3,
        )
        assert result.verdict in (QualityVerdict.WARNING, QualityVerdict.REJECT)

    def test_all_4_heads_run(self) -> None:
        """All 4 heads should produce results when historical data provided."""
        historical = np.array([12.0, 14.0, 16.0, 15.0, 13.0])
        result = detect_anomaly(
            c_obs=15.0,
            c_predicted=14.5,
            population_mean=15.0,
            population_sd=5.0,
            residual_sd=2.0,
            omega_cl=0.3,
            historical_concentrations=historical,
        )
        assert len(result.heads) == 4


# ══════════════════════════════════════════════════════════════════
# 3. SHAP Explainer
# ══════════════════════════════════════════════════════════════════

class TestSHAPExplainer:
    """Test SHAP explanations."""

    def test_explanation_has_features(self) -> None:
        """Explanation should have feature contributions."""
        patient = PatientData(
            age=60, weight=80, height=170,
            gender=Gender.MALE, serum_creatinine=1.5,
        )
        explanation = explain_pk_parameter(
            patient, crcl=50.0, pk_param="CL",
            base_value=3.52, predicted_value=2.8,
        )
        assert len(explanation.features) > 0
        assert explanation.parameter_name == "CL"
        assert explanation.base_value == 3.52

    def test_low_crcl_decreases_cl(self) -> None:
        """Low CrCL (50 vs ref 85) should decrease CL."""
        patient = PatientData(
            age=70, weight=65, height=165,
            gender=Gender.MALE, serum_creatinine=2.0,
        )
        explanation = explain_pk_parameter(
            patient, crcl=35.0, pk_param="CL",
            base_value=3.52, predicted_value=2.0,
        )
        # Find CrCL feature
        crcl_feat = next(
            (f for f in explanation.features if f.name == "crcl"), None
        )
        assert crcl_feat is not None
        assert crcl_feat.shap_value < 0  # Decreases CL

    def test_explanation_sorted_by_importance(self) -> None:
        """Features should be sorted by |SHAP| descending."""
        patient = PatientData(
            age=60, weight=80, height=170,
            gender=Gender.MALE, serum_creatinine=1.5,
        )
        explanation = explain_pk_parameter(
            patient, crcl=50.0, pk_param="CL",
            base_value=3.52, predicted_value=2.8,
        )
        shap_abs = [abs(f.shap_value) for f in explanation.features]
        assert shap_abs == sorted(shap_abs, reverse=True)


# ══════════════════════════════════════════════════════════════════
# 4. VPC Simulation (lightweight test)
# ══════════════════════════════════════════════════════════════════

class TestVPC:
    """Test VPC simulation (small N for speed)."""

    def test_vpc_generates_data(self) -> None:
        """VPC should produce percentile curves."""
        tv = PKParams(CL=3.5, V1=30.0, Q=5.0, V2=40.0)
        doses = [
            DoseEvent(time=0, amount=1000, duration=1.0,
                      route=Route.IV_INFUSION),
        ]
        vpc_data = simulate_vpc(
            tv, VANCOMYCIN_VN, doses,
            model_type=ModelType.TWO_COMP_IV,
            t_end=24.0, dt=1.0,
            n_simulations=20,   # Very small for testing
            n_replicates=5,
            seed=42,
        )

        assert len(vpc_data.time_grid) > 0
        assert len(vpc_data.sim_pctile_50) == len(vpc_data.time_grid)

    def test_vpc_percentile_ordering(self) -> None:
        """5th < 50th < 95th percentile at each time."""
        tv = PKParams(CL=3.5, V1=30.0, Q=5.0, V2=40.0)
        doses = [
            DoseEvent(time=0, amount=1000, duration=1.0,
                      route=Route.IV_INFUSION),
        ]
        vpc_data = simulate_vpc(
            tv, VANCOMYCIN_VN, doses,
            model_type=ModelType.TWO_COMP_IV,
            t_end=24.0, dt=2.0,
            n_simulations=50,
            n_replicates=3,
            seed=42,
        )

        # At non-zero time points, 5th < 50th < 95th
        for i in range(1, len(vpc_data.time_grid)):
            if vpc_data.sim_pctile_50[i] > 0:
                assert vpc_data.sim_pctile_5[i] <= vpc_data.sim_pctile_50[i]
                assert vpc_data.sim_pctile_50[i] <= vpc_data.sim_pctile_95[i]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
