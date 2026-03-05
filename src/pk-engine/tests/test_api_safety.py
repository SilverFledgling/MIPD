"""
Tests for FastAPI Routes + Safety Guardrails – Phase 7.

Tests:
    1. Safety guardrails (5 layers)
    2. FastAPI endpoint tests via TestClient
"""

import numpy as np
import pytest

from api.safety import (
    validate_patient, validate_doses, validate_observations,
    validate_pk_params, validate_recommended_dose,
    validate_confidence, generate_safety_report,
    AlertLevel, SafetyAlert,
)
from pk.models import (
    DoseEvent, Gender, PatientData, PKParams, Route,
)


# ══════════════════════════════════════════════════════════════════
# 1. Safety Guardrails – Layer 1: Input Validation
# ══════════════════════════════════════════════════════════════════

class TestSafetyInputValidation:
    """Test input validation for patient demographics, doses, TDM."""

    def test_valid_patient_no_alerts(self) -> None:
        """Normal patient should generate no alerts."""
        patient = PatientData(
            age=55, weight=70, height=170,
            gender=Gender.MALE, serum_creatinine=1.2,
        )
        alerts = validate_patient(patient)
        assert len(alerts) == 0

    def test_extreme_age_flagged(self) -> None:
        """Age outside 18-100 should be flagged."""
        patient = PatientData(
            age=15, weight=70, height=170,
            gender=Gender.MALE, serum_creatinine=1.2,
        )
        alerts = validate_patient(patient)
        assert any(a.code == "PATIENT_AGE_OUT_OF_RANGE" for a in alerts)

    def test_extreme_weight_flagged(self) -> None:
        """Weight below 30kg should be critical."""
        patient = PatientData(
            age=55, weight=25, height=170,
            gender=Gender.MALE, serum_creatinine=1.2,
        )
        alerts = validate_patient(patient)
        assert any(a.level == AlertLevel.CRITICAL for a in alerts)

    def test_high_scr_flagged(self) -> None:
        """SCr > 15 mg/dL should be flagged."""
        patient = PatientData(
            age=55, weight=70, height=170,
            gender=Gender.MALE, serum_creatinine=16.0,
        )
        alerts = validate_patient(patient)
        assert any(a.code == "SCR_EXTREMELY_HIGH" for a in alerts)

    def test_valid_doses_no_alerts(self) -> None:
        """Normal doses should be fine."""
        doses = [
            DoseEvent(time=0, amount=1000, duration=1.0, route=Route.IV_INFUSION),
        ]
        alerts = validate_doses(doses)
        assert len(alerts) == 0

    def test_excessive_dose_flagged(self) -> None:
        """Dose > 3000mg should be critical."""
        doses = [
            DoseEvent(time=0, amount=5000, duration=1.0, route=Route.IV_INFUSION),
        ]
        alerts = validate_doses(doses)
        assert any(a.code == "DOSE_EXCEEDS_MAXIMUM" for a in alerts)

    def test_fast_infusion_rate_flagged(self) -> None:
        """Too-fast infusion (Red Man Syndrome risk) should be flagged."""
        doses = [
            DoseEvent(time=0, amount=2000, duration=0.5, route=Route.IV_INFUSION),
        ]
        alerts = validate_doses(doses)
        assert any(a.code == "INFUSION_RATE_TOO_FAST" for a in alerts)

    def test_no_doses_rejected(self) -> None:
        """No doses should be rejected."""
        alerts = validate_doses([])
        assert any(a.level == AlertLevel.REJECT for a in alerts)

    def test_valid_observation_no_alerts(self) -> None:
        """Normal TDM value should be fine."""
        alerts = validate_observations([
            {"concentration": 15.0, "time": 12.0},
        ])
        assert len(alerts) == 0

    def test_extreme_concentration_flagged(self) -> None:
        """Concentration > 100 mg/L should be flagged."""
        alerts = validate_observations([
            {"concentration": 150.0, "time": 12.0},
        ])
        assert any(a.code == "CONC_ABOVE_ASSAY_LIMIT" for a in alerts)


# ══════════════════════════════════════════════════════════════════
# 2. Safety Guardrails – Layer 2: PK Parameter Plausibility
# ══════════════════════════════════════════════════════════════════

class TestSafetyPKPlausibility:
    """Test PK parameter plausibility checking."""

    def test_normal_params_no_alerts(self) -> None:
        """Normal PK params should be fine."""
        params = PKParams(CL=3.5, V1=35.0, Q=4.0, V2=40.0)
        alerts = validate_pk_params(params)
        assert len(alerts) == 0

    def test_low_cl_flagged(self) -> None:
        """CL < 0.5 L/h should be flagged."""
        params = PKParams(CL=0.1, V1=35.0, Q=4.0, V2=40.0)
        alerts = validate_pk_params(params)
        assert any(a.code == "PK_CL_TOO_LOW" for a in alerts)

    def test_high_v1_flagged(self) -> None:
        """V1 > 100 L should be flagged."""
        params = PKParams(CL=3.5, V1=150.0, Q=4.0, V2=40.0)
        alerts = validate_pk_params(params)
        assert any(a.code == "PK_V1_TOO_HIGH" for a in alerts)


# ══════════════════════════════════════════════════════════════════
# 3. Safety Guardrails – Layer 3: Dose Limits
# ══════════════════════════════════════════════════════════════════

class TestSafetyDoseLimits:
    """Test dose recommendation limits."""

    def test_safe_dose_recommendation(self) -> None:
        """Normal dose should pass."""
        alerts = validate_recommended_dose(1000, current_dose=1000)
        assert len(alerts) == 0

    def test_excessive_dose_rejected(self) -> None:
        """Dose > max should be rejected."""
        alerts = validate_recommended_dose(5000)
        assert any(a.level == AlertLevel.REJECT for a in alerts)

    def test_large_dose_change_warned(self) -> None:
        """Dose change > 50% should generate warning."""
        alerts = validate_recommended_dose(2000, current_dose=1000)
        assert any(a.code == "LARGE_DOSE_CHANGE" for a in alerts)


# ══════════════════════════════════════════════════════════════════
# 4. Safety Guardrails – Layer 4: Confidence
# ══════════════════════════════════════════════════════════════════

class TestSafetyConfidence:
    """Test confidence interval width validation."""

    def test_narrow_ci_ok(self) -> None:
        """Tight CI should pass."""
        alerts = validate_confidence(2.0, 5.0, "CL")
        assert len(alerts) == 0

    def test_wide_ci_flagged(self) -> None:
        """CI ratio > 3 should be flagged."""
        alerts = validate_confidence(1.0, 10.0, "CL", max_ci_ratio=3.0)
        assert any(a.code == "HIGH_UNCERTAINTY" for a in alerts)


# ══════════════════════════════════════════════════════════════════
# 5. Safety Report Generation
# ══════════════════════════════════════════════════════════════════

class TestSafetyReport:
    """Test consolidated safety report."""

    def test_no_alerts_safe(self) -> None:
        """No alerts = safe."""
        report = generate_safety_report([])
        assert report.is_safe
        assert report.risk_score == 0.0

    def test_warning_gives_moderate_risk(self) -> None:
        """Warning should give risk_score = 0.5."""
        alerts = [SafetyAlert(AlertLevel.WARNING, "TEST", "test")]
        report = generate_safety_report(alerts)
        assert report.risk_score == 0.5
        assert report.is_safe  # Warnings don't block

    def test_critical_gives_high_risk(self) -> None:
        """Critical should give risk_score = 0.75, not safe."""
        alerts = [SafetyAlert(AlertLevel.CRITICAL, "TEST", "test")]
        report = generate_safety_report(alerts)
        assert report.risk_score == 0.75
        assert not report.is_safe
        assert report.requires_review

    def test_reject_gives_max_risk(self) -> None:
        """Reject should give risk_score = 1.0."""
        alerts = [SafetyAlert(AlertLevel.REJECT, "TEST", "test")]
        report = generate_safety_report(alerts)
        assert report.risk_score == 1.0
        assert not report.is_safe


# ══════════════════════════════════════════════════════════════════
# 6. FastAPI Endpoints (if fastapi installed)
# ══════════════════════════════════════════════════════════════════

try:
    from fastapi.testclient import TestClient
    from api.main import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
class TestFastAPIEndpoints:
    """Test actual API endpoints via TestClient."""

    def _client(self):
        return TestClient(app)

    def test_health_check(self) -> None:
        """Health endpoint should return 200."""
        resp = self._client().get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_safety_info(self) -> None:
        """/safety-info should describe all 5 layers."""
        resp = self._client().get("/safety-info")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["layers"]) == 5

    def test_pk_predict(self) -> None:
        """/pk/predict should return concentrations."""
        resp = self._client().post("/pk/predict", json={
            "patient": {
                "age": 55, "weight": 70, "height": 170,
                "gender": "male", "serum_creatinine": 1.2,
            },
            "doses": [
                {"time": 0, "amount": 1000, "duration": 1.0,
                 "route": "iv_infusion"},
            ],
            "times": [1, 6, 12],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["concentrations"]) == 3
        assert all(c > 0 for c in data["concentrations"])
        assert "safety" in data

    def test_pk_clinical(self) -> None:
        """/pk/clinical should return clinical calculations."""
        resp = self._client().post("/pk/clinical", json={
            "patient": {
                "age": 55, "weight": 70, "height": 170,
                "gender": "male", "serum_creatinine": 1.2,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["crcl_ml_min"] > 0
        assert data["ibw_kg"] > 0

    def test_bayesian_map(self) -> None:
        """/bayesian/estimate MAP should return PK params."""
        resp = self._client().post("/bayesian/estimate", json={
            "patient": {
                "age": 55, "weight": 70, "height": 170,
                "gender": "male", "serum_creatinine": 1.2,
            },
            "doses": [
                {"time": 0, "amount": 1000, "duration": 1.0,
                 "route": "iv_infusion"},
            ],
            "observations": [
                {"time": 11.5, "concentration": 12.0,
                 "sample_type": "trough"},
            ],
            "method": "map",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["individual_params"]["CL"] > 0
        assert data["method"] == "map"
        assert "safety" in data

    def test_anomaly_check(self) -> None:
        """/ai/anomaly-check should return quality score."""
        resp = self._client().post("/ai/anomaly-check", json={
            "patient": {
                "age": 55, "weight": 70, "height": 170,
                "gender": "male", "serum_creatinine": 1.2,
            },
            "doses": [
                {"time": 0, "amount": 1000, "duration": 1.0,
                 "route": "iv_infusion"},
            ],
            "observations": [
                {"time": 11.5, "concentration": 12.0,
                 "sample_type": "trough"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["quality_score"] <= 1
        assert data["verdict"] in ["ACCEPT", "WARNING", "REJECT"]

    def test_invalid_patient_rejected(self) -> None:
        """Invalid patient data should return 422."""
        resp = self._client().post("/pk/predict", json={
            "patient": {
                "age": -5, "weight": 70, "height": 170,
                "gender": "male", "serum_creatinine": 1.2,
            },
            "doses": [{"time": 0, "amount": 1000, "duration": 1.0}],
            "times": [1],
        })
        assert resp.status_code == 422  # Pydantic validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
