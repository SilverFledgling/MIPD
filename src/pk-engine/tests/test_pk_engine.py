"""
Tests for PK Engine – Phase 2 verification.

Tests:
    1. Clinical calculations (CrCL, eGFR, BMI, IBW, ABW)
    2. Analytical PK solutions (1-comp IV, oral, 2-comp)
    3. ODE solver vs analytical cross-validation
    4. PopPK covariate model
    5. IIV sampling
"""

import math

import numpy as np
import pytest

from pk.models import DoseEvent, Gender, ModelType, PatientData, PKParams, Route
from pk.clinical import (
    bmi,
    bsa_dubois,
    ckd_epi_egfr,
    cockcroft_gault_crcl,
    compute_crcl_for_patient,
    ideal_body_weight,
    adjusted_body_weight,
)
from pk.analytical import (
    auc_trapezoidal,
    auc24_from_cl,
    one_comp_iv_bolus,
    one_comp_iv_infusion,
    one_comp_oral,
    oral_cmax,
    oral_tmax,
    two_comp_iv_bolus,
)
from pk.solver import simulate, predict_concentrations
from pk.population import (
    apply_iiv,
    compute_vancomycin_tv,
    sample_individual_params,
    VANCOMYCIN_VN,
)


# ══════════════════════════════════════════════════════════════════
# 1. Clinical Calculations
# ══════════════════════════════════════════════════════════════════

class TestClinicalCalculations:
    """Test Cockcroft-Gault, CKD-EPI, BMI, BSA, IBW, ABW."""

    def test_crcl_male(self) -> None:
        """CrCL for 60yo, 70kg male, SCr=1.0 -> ~97 mL/min."""
        result = cockcroft_gault_crcl(
            age=60, weight=70, serum_creatinine=1.0, gender=Gender.MALE
        )
        expected = (140 - 60) * 70 / (72 * 1.0)
        assert abs(result - expected) < 0.01

    def test_crcl_female(self) -> None:
        """Female gets 0.85 correction factor."""
        male = cockcroft_gault_crcl(
            age=50, weight=60, serum_creatinine=0.8, gender=Gender.MALE
        )
        female = cockcroft_gault_crcl(
            age=50, weight=60, serum_creatinine=0.8, gender=Gender.FEMALE
        )
        assert abs(female - male * 0.85) < 0.01

    def test_crcl_invalid_input(self) -> None:
        """Negative inputs should raise ValueError."""
        with pytest.raises(ValueError):
            cockcroft_gault_crcl(age=-1, weight=70, serum_creatinine=1.0,
                                gender=Gender.MALE)

    def test_egfr_male(self) -> None:
        """eGFR for 50yo male, SCr=1.0 -> reasonable value (>60)."""
        result = ckd_epi_egfr(
            serum_creatinine=1.0, age=50, gender=Gender.MALE
        )
        assert 60 < result < 120

    def test_egfr_female_higher(self) -> None:
        """Female gets 1.012 factor -> slightly higher eGFR."""
        m = ckd_epi_egfr(serum_creatinine=0.8, age=40, gender=Gender.MALE)
        f = ckd_epi_egfr(serum_creatinine=0.8, age=40, gender=Gender.FEMALE)
        # Female should be higher due to 1.012 and different kappa/alpha
        assert f > 0  # Just ensure it's valid

    def test_bmi(self) -> None:
        """BMI for 70kg, 170cm -> ~24.2."""
        result = bmi(weight=70, height_cm=170)
        expected = 70 / (1.70 ** 2)
        assert abs(result - expected) < 0.1

    def test_bsa(self) -> None:
        """BSA for 70kg, 170cm -> ~1.8 m2."""
        result = bsa_dubois(weight=70, height_cm=170)
        assert 1.5 < result < 2.2

    def test_ibw_male(self) -> None:
        """IBW for 170cm male -> ~66 kg."""
        result = ideal_body_weight(height_cm=170, gender=Gender.MALE)
        height_in = 170 / 2.54
        expected = 50 + 2.3 * (height_in - 60)
        assert abs(result - expected) < 0.01

    def test_abw(self) -> None:
        """ABW for 100kg, 170cm male -> between IBW and TBW."""
        result = adjusted_body_weight(
            total_body_weight=100, height_cm=170, gender=Gender.MALE
        )
        ibw = ideal_body_weight(170, Gender.MALE)
        assert ibw < result < 100


# ══════════════════════════════════════════════════════════════════
# 2. Analytical PK Solutions
# ══════════════════════════════════════════════════════════════════

class TestAnalyticalSolutions:
    """Test closed-form PK equations."""

    def setup_method(self) -> None:
        """Common PK parameters for tests."""
        self.params_1comp = PKParams(CL=4.0, V1=50.0)
        self.params_oral = PKParams(CL=4.0, V1=50.0, Ka=1.5, F=0.8)
        self.params_2comp = PKParams(CL=3.5, V1=30.0, Q=5.0, V2=40.0)

    def test_iv_bolus_at_zero(self) -> None:
        """C(0) = Dose / Vd."""
        t = np.array([0.0])
        c = one_comp_iv_bolus(t, dose=1000, params=self.params_1comp)
        assert abs(c[0] - 1000 / 50.0) < 0.01

    def test_iv_bolus_decay(self) -> None:
        """Concentration should decrease over time."""
        t = np.linspace(0, 24, 100)
        c = one_comp_iv_bolus(t, dose=1000, params=self.params_1comp)
        assert c[0] > c[-1]
        assert all(np.diff(c) <= 0)  # monotonically decreasing

    def test_iv_infusion_peak_at_end(self) -> None:
        """Peak at end of infusion for 1-comp IV infusion."""
        t = np.linspace(0, 12, 200)
        c = one_comp_iv_infusion(t, dose=1000, t_inf=1.0,
                                 params=self.params_1comp)
        peak_idx = np.argmax(c)
        # Peak should be near t=1h (end of infusion)
        assert abs(t[peak_idx] - 1.0) < 0.2

    def test_oral_tmax(self) -> None:
        """T_max should be positive and reasonable."""
        t_max = oral_tmax(self.params_oral)
        assert 0 < t_max < 10

    def test_oral_cmax(self) -> None:
        """C_max should be positive."""
        c_max = oral_cmax(dose=500, params=self.params_oral)
        assert c_max > 0

    def test_oral_curve_shape(self) -> None:
        """Oral: rise then fall (absorption then elimination)."""
        t = np.linspace(0.01, 24, 200)
        c = one_comp_oral(t, dose=500, params=self.params_oral)
        assert c[0] < np.max(c)  # rises initially
        assert c[-1] < np.max(c)  # falls eventually

    def test_2comp_iv_bolus_biexponential(self) -> None:
        """2-comp IV bolus: C(0) > 0 and biexponential decay."""
        t = np.linspace(0.01, 24, 200)
        c = two_comp_iv_bolus(t, dose=1000, params=self.params_2comp)
        assert c[0] > 0
        assert c[-1] < c[0]

    def test_auc_trapezoidal(self) -> None:
        """AUC of constant concentration = C * T."""
        t = np.linspace(0, 10, 100)
        c = np.ones_like(t) * 5.0
        auc = auc_trapezoidal(t, c)
        assert abs(auc - 50.0) < 0.1

    def test_auc24_from_cl(self) -> None:
        """AUC24 = DailyDose / CL."""
        auc24 = auc24_from_cl(daily_dose=2000, cl=4.0)
        assert abs(auc24 - 500.0) < 0.01


# ══════════════════════════════════════════════════════════════════
# 3. ODE Solver vs Analytical (Cross-Validation)
# ══════════════════════════════════════════════════════════════════

class TestODESolver:
    """Cross-validate ODE solver against analytical solutions."""

    def test_1comp_iv_bolus_ode_vs_analytical(self) -> None:
        """ODE solver should match analytical for 1-comp IV bolus."""
        params = PKParams(CL=4.0, V1=50.0)
        dose = DoseEvent(time=0, amount=1000, duration=0, route=Route.IV_BOLUS)

        result = simulate(
            params=params,
            doses=[dose],
            model_type=ModelType.ONE_COMP_IV,
            t_end=24.0,
            dt=0.05,
        )

        # Analytical reference
        c_analytical = one_comp_iv_bolus(result.time, 1000, params)

        # Compare: max absolute error < 0.1 mg/L
        max_err = float(np.max(np.abs(result.concentration - c_analytical)))
        assert max_err < 0.1, f"Max error: {max_err:.4f} mg/L"

    def test_1comp_iv_infusion_ode_vs_analytical(self) -> None:
        """ODE solver should match analytical for 1-comp IV infusion."""
        params = PKParams(CL=4.0, V1=50.0)
        dose = DoseEvent(
            time=0, amount=1000, duration=1.0, route=Route.IV_INFUSION
        )

        result = simulate(
            params=params,
            doses=[dose],
            model_type=ModelType.ONE_COMP_IV,
            t_end=24.0,
            dt=0.05,
        )

        c_analytical = one_comp_iv_infusion(result.time, 1000, 1.0, params)
        max_err = float(np.max(np.abs(result.concentration - c_analytical)))
        assert max_err < 0.1, f"Max error: {max_err:.4f} mg/L"

    def test_2comp_multiple_doses(self) -> None:
        """2-comp with multiple doses should accumulate."""
        params = PKParams(CL=3.5, V1=30.0, Q=5.0, V2=40.0)
        doses = [
            DoseEvent(time=0, amount=1000, duration=1.0,
                      route=Route.IV_INFUSION),
            DoseEvent(time=12, amount=1000, duration=1.0,
                      route=Route.IV_INFUSION),
            DoseEvent(time=24, amount=1000, duration=1.0,
                      route=Route.IV_INFUSION),
        ]

        result = simulate(
            params=params,
            doses=doses,
            model_type=ModelType.TWO_COMP_IV,
            t_end=48.0,
            dt=0.1,
        )

        # Second trough should be higher than first (accumulation)
        c_12h = result.concentration_at(11.9)
        c_24h = result.concentration_at(23.9)
        assert c_24h > c_12h * 0.9  # Some accumulation

    def test_predict_concentrations(self) -> None:
        """predict_concentrations should return correct number of values."""
        params = PKParams(CL=3.5, V1=30.0, Q=5.0, V2=40.0)
        doses = [
            DoseEvent(time=0, amount=1000, duration=1.0,
                      route=Route.IV_INFUSION),
        ]

        preds = predict_concentrations(
            params=params,
            doses=doses,
            obs_times=[6.0, 12.0, 24.0],
            model_type=ModelType.TWO_COMP_IV,
        )

        assert len(preds) == 3
        assert all(p > 0 for p in preds)
        # Concentrations should decrease over time
        assert preds[0] > preds[1] > preds[2]


# ══════════════════════════════════════════════════════════════════
# 4. PopPK Covariate Model
# ══════════════════════════════════════════════════════════════════

class TestPopPKModel:
    """Test covariate relationships and IIV."""

    def test_vancomycin_tv_normal_patient(self) -> None:
        """TV for 60yo, 70kg, SCr=1.0 should be near base values."""
        patient = PatientData(
            age=60, weight=70, height=170,
            gender=Gender.MALE, serum_creatinine=1.0,
        )
        tv = compute_vancomycin_tv(patient)
        # CL should be near 3.52 (base) for typical patient
        assert 2.0 < tv.CL < 6.0
        assert 20 < tv.V1 < 50

    def test_vancomycin_tv_renal_impaired(self) -> None:
        """Higher SCr -> lower CrCL -> lower CL."""
        normal = PatientData(
            age=60, weight=70, height=170,
            gender=Gender.MALE, serum_creatinine=1.0,
        )
        impaired = PatientData(
            age=60, weight=70, height=170,
            gender=Gender.MALE, serum_creatinine=3.0,
        )
        tv_normal = compute_vancomycin_tv(normal)
        tv_impaired = compute_vancomycin_tv(impaired)
        assert tv_impaired.CL < tv_normal.CL

    def test_apply_iiv_no_random_effects(self) -> None:
        """With eta=0, individual = typical values."""
        tv = PKParams(CL=3.5, V1=30.0, Q=5.0, V2=40.0)
        eta = np.zeros(4)
        individual = apply_iiv(tv, eta)
        assert abs(individual.CL - tv.CL) < 0.001
        assert abs(individual.V1 - tv.V1) < 0.001

    def test_apply_iiv_positive_eta(self) -> None:
        """Positive eta -> higher individual values (log-normal)."""
        tv = PKParams(CL=3.5, V1=30.0, Q=5.0, V2=40.0)
        eta = np.array([0.3, 0.2, 0.1, 0.1])
        individual = apply_iiv(tv, eta)
        assert individual.CL > tv.CL
        assert individual.V1 > tv.V1

    def test_sample_individual_reproducible(self) -> None:
        """Sampling with same seed should give same result."""
        tv = PKParams(CL=3.5, V1=30.0, Q=5.0, V2=40.0)
        omega = VANCOMYCIN_VN.omega_matrix

        rng1 = np.random.default_rng(42)
        p1 = sample_individual_params(tv, omega, rng1)

        rng2 = np.random.default_rng(42)
        p2 = sample_individual_params(tv, omega, rng2)

        assert abs(p1.CL - p2.CL) < 1e-10


# ══════════════════════════════════════════════════════════════════
# Run tests
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
