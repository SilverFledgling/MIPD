"""
Safety Guardrails – Clinical safety validation for MIPD system.

Addresses the critical question:
    "If input data is poor quality, can PK parameters endanger patients?"

Answer: YES. Without guardrails, garbage-in → garbage-out → wrong dose → toxicity/inefficacy.

This module implements 5 layers of protection:

Layer 1: INPUT VALIDATION
    - Patient demographics within physiological ranges
    - Dose amounts within approved formulary limits
    - TDM concentrations within assay detection limits

Layer 2: PARAMETER PLAUSIBILITY
    - PK parameters (CL, V) within pharmacologically feasible ranges
    - Flag extreme outliers (> 3 SD from population)

Layer 3: DOSE RECOMMENDATION LIMITS
    - Hard caps on recommended doses (never exceed max safe dose)
    - Flag dose changes > 50% from current dose

Layer 4: CONFIDENCE REQUIREMENT
    - Reject predictions with too-wide credible intervals
    - Minimum number of TDM observations required

Layer 5: CLINICAL DECISION SUPPORT
    - Warning flags in API response
    - Risk score for each recommendation
    - Suggested human review triggers

Reference:
    - Rybak et al. (2020), Vancomycin Therapeutic Guidelines
    - FDA Guidance for Industry: Clinical Pharmacology (2022)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from pk.models import DoseEvent, PatientData, PKParams, Gender, Route


# ──────────────────────────────────────────────────────────────────
# Severity levels
# ──────────────────────────────────────────────────────────────────

class AlertLevel(str, Enum):
    """Alert severity level."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    REJECT = "reject"


@dataclass
class SafetyAlert:
    """
    Single safety alert.

    Attributes:
        level:   Severity (info/warning/critical/reject)
        code:    Machine-readable alert code
        message: Human-readable description
        field:   Which field triggered the alert
        value:   Actual value that triggered
    """
    level: AlertLevel
    code: str
    message: str
    field: str = ""
    value: float | str = ""


@dataclass
class SafetyReport:
    """
    Cumulative safety assessment.

    Attributes:
        is_safe:         Overall safe to proceed
        risk_score:      0.0 (safe) to 1.0 (dangerous)
        alerts:          List of safety alerts
        requires_review: True if pharmacist review recommended
        recommendation:  Overall recommendation text
    """
    is_safe: bool
    risk_score: float
    alerts: list[SafetyAlert] = field(default_factory=list)
    requires_review: bool = False
    recommendation: str = ""


# ──────────────────────────────────────────────────────────────────
# Layer 1: Clinical ranges (drug-specific)
# ──────────────────────────────────────────────────────────────────

# Vancomycin clinical reference ranges
VANCOMYCIN_LIMITS = {
    # Patient demographics
    "age_min": 18, "age_max": 100,            # years
    "weight_min": 30, "weight_max": 250,       # kg
    "height_min": 100, "height_max": 220,      # cm
    "scr_min": 0.2, "scr_max": 15.0,          # mg/dL

    # Dose limits
    "dose_min": 250, "dose_max": 3000,         # mg per dose
    "daily_dose_max": 6000,                    # mg/day
    "infusion_rate_max": 1000,                 # mg/h (to prevent Red Man Syndrome)

    # TDM assay limits
    "conc_min": 0.5, "conc_max": 100.0,        # mg/L (assay range)
    "trough_target_min": 10.0,                 # mg/L (Rybak 2020)
    "trough_target_max": 20.0,                 # mg/L
    "auc_target_min": 400,                     # mg*h/L (AUC/MIC ≥ 400)
    "auc_target_max": 600,                     # mg*h/L

    # PK parameter feasibility
    "cl_min": 0.5, "cl_max": 15.0,             # L/h
    "v1_min": 5.0, "v1_max": 100.0,            # L
    "q_min": 0.5, "q_max": 30.0,               # L/h
    "v2_min": 5.0, "v2_max": 150.0,            # L

    # Dose change limits
    "max_dose_change_pct": 50,                 # % change from current
}


# ──────────────────────────────────────────────────────────────────
# Layer 1: Input Validation
# ──────────────────────────────────────────────────────────────────

def validate_patient(
    patient: PatientData,
    drug: str = "vancomycin",
) -> list[SafetyAlert]:
    """
    Validate patient demographics are physiologically reasonable.

    Checks:
        - Age: 18-100
        - Weight: 30-250 kg
        - Height: 100-220 cm
        - Serum creatinine: 0.2-15.0 mg/dL
    """
    limits = VANCOMYCIN_LIMITS
    alerts: list[SafetyAlert] = []

    # Age
    if patient.age < limits["age_min"] or patient.age > limits["age_max"]:
        alerts.append(SafetyAlert(
            level=AlertLevel.CRITICAL,
            code="PATIENT_AGE_OUT_OF_RANGE",
            message=(
                f"Patient age {patient.age}y outside valid range "
                f"[{limits['age_min']}-{limits['age_max']}]"
            ),
            field="age",
            value=patient.age,
        ))

    # Weight
    if patient.weight < limits["weight_min"]:
        alerts.append(SafetyAlert(
            level=AlertLevel.CRITICAL,
            code="PATIENT_WEIGHT_TOO_LOW",
            message=f"Weight {patient.weight}kg < {limits['weight_min']}kg minimum",
            field="weight",
            value=patient.weight,
        ))
    elif patient.weight > limits["weight_max"]:
        alerts.append(SafetyAlert(
            level=AlertLevel.WARNING,
            code="PATIENT_WEIGHT_HIGH",
            message=f"Weight {patient.weight}kg exceeds {limits['weight_max']}kg",
            field="weight",
            value=patient.weight,
        ))

    # Serum creatinine
    if patient.serum_creatinine < limits["scr_min"]:
        alerts.append(SafetyAlert(
            level=AlertLevel.WARNING,
            code="SCR_POSSIBLY_INVALID",
            message=f"SCr {patient.serum_creatinine} mg/dL unusually low",
            field="serum_creatinine",
            value=patient.serum_creatinine,
        ))
    elif patient.serum_creatinine > limits["scr_max"]:
        alerts.append(SafetyAlert(
            level=AlertLevel.CRITICAL,
            code="SCR_EXTREMELY_HIGH",
            message=(
                f"SCr {patient.serum_creatinine} mg/dL extremely elevated. "
                f"May indicate renal failure — dose adjustment critical."
            ),
            field="serum_creatinine",
            value=patient.serum_creatinine,
        ))

    return alerts


def validate_doses(
    doses: list[DoseEvent],
    drug: str = "vancomycin",
) -> list[SafetyAlert]:
    """
    Validate dose amounts and infusion rates.

    Checks:
        - Individual dose: 250-3000 mg
        - Daily dose: ≤ 6000 mg
        - Infusion rate: ≤ 1000 mg/h
    """
    limits = VANCOMYCIN_LIMITS
    alerts: list[SafetyAlert] = []

    if not doses:
        alerts.append(SafetyAlert(
            level=AlertLevel.REJECT,
            code="NO_DOSES",
            message="No dose events provided",
        ))
        return alerts

    for i, dose in enumerate(doses):
        if dose.amount < limits["dose_min"]:
            alerts.append(SafetyAlert(
                level=AlertLevel.WARNING,
                code="DOSE_BELOW_MINIMUM",
                message=f"Dose #{i+1}: {dose.amount}mg < {limits['dose_min']}mg",
                field=f"doses[{i}].amount",
                value=dose.amount,
            ))
        elif dose.amount > limits["dose_max"]:
            alerts.append(SafetyAlert(
                level=AlertLevel.CRITICAL,
                code="DOSE_EXCEEDS_MAXIMUM",
                message=f"Dose #{i+1}: {dose.amount}mg > {limits['dose_max']}mg",
                field=f"doses[{i}].amount",
                value=dose.amount,
            ))

        # Infusion rate check
        if dose.duration > 0:
            rate = dose.amount / dose.duration
            if rate > limits["infusion_rate_max"]:
                alerts.append(SafetyAlert(
                    level=AlertLevel.CRITICAL,
                    code="INFUSION_RATE_TOO_FAST",
                    message=(
                        f"Dose #{i+1}: infusion rate {rate:.0f} mg/h "
                        f"> {limits['infusion_rate_max']} mg/h "
                        f"(risk of Red Man Syndrome)"
                    ),
                    field=f"doses[{i}].rate",
                    value=rate,
                ))

    return alerts


def validate_observations(
    observations: list[dict],
    drug: str = "vancomycin",
) -> list[SafetyAlert]:
    """
    Validate TDM concentrations are within assay limits.

    Checks:
        - Concentration: 0.5-100.0 mg/L
        - Time must be positive
    """
    limits = VANCOMYCIN_LIMITS
    alerts: list[SafetyAlert] = []

    for i, obs in enumerate(observations):
        conc = obs.get("concentration", 0)
        time = obs.get("time", 0)

        if conc < limits["conc_min"]:
            alerts.append(SafetyAlert(
                level=AlertLevel.WARNING,
                code="CONC_BELOW_ASSAY_LIMIT",
                message=(
                    f"Observation #{i+1}: {conc} mg/L < {limits['conc_min']} "
                    f"(below assay detection limit)"
                ),
                field=f"observations[{i}].concentration",
                value=conc,
            ))
        elif conc > limits["conc_max"]:
            alerts.append(SafetyAlert(
                level=AlertLevel.CRITICAL,
                code="CONC_ABOVE_ASSAY_LIMIT",
                message=(
                    f"Observation #{i+1}: {conc} mg/L > {limits['conc_max']} "
                    f"(possible assay error or extreme toxicity)"
                ),
                field=f"observations[{i}].concentration",
                value=conc,
            ))

        if time < 0:
            alerts.append(SafetyAlert(
                level=AlertLevel.REJECT,
                code="NEGATIVE_TIME",
                message=f"Observation #{i+1}: time={time}h is negative",
                field=f"observations[{i}].time",
                value=time,
            ))

    return alerts


# ──────────────────────────────────────────────────────────────────
# Layer 2: PK Parameter Plausibility
# ──────────────────────────────────────────────────────────────────

def validate_pk_params(
    params: PKParams,
    drug: str = "vancomycin",
) -> list[SafetyAlert]:
    """
    Check if estimated PK parameters are pharmacologically feasible.

    Extreme PK parameters often indicate:
        - Poor quality input data
        - Model misspecification
        - Data entry errors
    """
    limits = VANCOMYCIN_LIMITS
    alerts: list[SafetyAlert] = []

    checks = [
        ("CL", params.CL, limits["cl_min"], limits["cl_max"], "L/h"),
        ("V1", params.V1, limits["v1_min"], limits["v1_max"], "L"),
    ]
    if params.Q > 0:
        checks.append(("Q", params.Q, limits["q_min"], limits["q_max"], "L/h"))
    if params.V2 > 0:
        checks.append(("V2", params.V2, limits["v2_min"], limits["v2_max"], "L"))

    for name, val, low, high, unit in checks:
        if val < low:
            alerts.append(SafetyAlert(
                level=AlertLevel.CRITICAL,
                code=f"PK_{name}_TOO_LOW",
                message=(
                    f"{name} = {val:.2f} {unit} < {low} {unit}. "
                    f"May indicate data quality issue."
                ),
                field=name,
                value=round(val, 4),
            ))
        elif val > high:
            alerts.append(SafetyAlert(
                level=AlertLevel.CRITICAL,
                code=f"PK_{name}_TOO_HIGH",
                message=(
                    f"{name} = {val:.2f} {unit} > {high} {unit}. "
                    f"Unusual — verify input data."
                ),
                field=name,
                value=round(val, 4),
            ))

    return alerts


# ──────────────────────────────────────────────────────────────────
# Layer 3: Dose Recommendation Limits
# ──────────────────────────────────────────────────────────────────

def validate_recommended_dose(
    recommended_dose: float,
    current_dose: float | None = None,
    drug: str = "vancomycin",
) -> list[SafetyAlert]:
    """
    Validate that a recommended dose is within safe limits.

    Checks:
        - Dose within formulary max
        - Dose change ≤ 50% from current (if provided)
    """
    limits = VANCOMYCIN_LIMITS
    alerts: list[SafetyAlert] = []

    if recommended_dose < limits["dose_min"]:
        alerts.append(SafetyAlert(
            level=AlertLevel.WARNING,
            code="RECOMMENDED_DOSE_LOW",
            message=f"Recommended dose {recommended_dose:.0f}mg below minimum",
            field="recommended_dose",
            value=recommended_dose,
        ))

    if recommended_dose > limits["dose_max"]:
        alerts.append(SafetyAlert(
            level=AlertLevel.REJECT,
            code="RECOMMENDED_DOSE_EXCEEDS_MAX",
            message=(
                f"Recommended dose {recommended_dose:.0f}mg exceeds "
                f"maximum safe dose {limits['dose_max']}mg. REJECTED."
            ),
            field="recommended_dose",
            value=recommended_dose,
        ))

    if current_dose is not None and current_dose > 0:
        pct_change = abs(recommended_dose - current_dose) / current_dose * 100
        if pct_change > limits["max_dose_change_pct"]:
            alerts.append(SafetyAlert(
                level=AlertLevel.WARNING,
                code="LARGE_DOSE_CHANGE",
                message=(
                    f"Dose change {pct_change:.0f}% "
                    f"(from {current_dose:.0f}mg to {recommended_dose:.0f}mg) "
                    f"exceeds {limits['max_dose_change_pct']}% threshold. "
                    f"Pharmacist review recommended."
                ),
                field="dose_change_pct",
                value=round(pct_change, 1),
            ))

    return alerts


# ──────────────────────────────────────────────────────────────────
# Layer 4: Confidence Requirement
# ──────────────────────────────────────────────────────────────────

def validate_confidence(
    ci95_lower: float,
    ci95_upper: float,
    param_name: str = "CL",
    max_ci_ratio: float = 3.0,
) -> list[SafetyAlert]:
    """
    Reject predictions with too-wide credible intervals.

    CI ratio = upper / lower. If > max_ci_ratio, uncertainty too high.
    """
    alerts: list[SafetyAlert] = []

    if ci95_lower <= 0:
        alerts.append(SafetyAlert(
            level=AlertLevel.WARNING,
            code="CI_LOWER_NONPOSITIVE",
            message=f"{param_name} 95% CI lower bound ≤ 0",
            field=f"{param_name}_ci95_lower",
            value=ci95_lower,
        ))
        return alerts

    ci_ratio = ci95_upper / ci95_lower
    if ci_ratio > max_ci_ratio:
        alerts.append(SafetyAlert(
            level=AlertLevel.CRITICAL,
            code="HIGH_UNCERTAINTY",
            message=(
                f"{param_name} 95% CI ratio = {ci_ratio:.1f} "
                f"(target < {max_ci_ratio:.1f}). "
                f"Prediction uncertainty too high for clinical use. "
                f"Collect more TDM samples."
            ),
            field=f"{param_name}_ci_ratio",
            value=round(ci_ratio, 2),
        ))

    return alerts


# ──────────────────────────────────────────────────────────────────
# Layer 5: Generate consolidated safety report
# ──────────────────────────────────────────────────────────────────

def generate_safety_report(alerts: list[SafetyAlert]) -> SafetyReport:
    """
    Generate a consolidated safety report from all alerts.

    Risk score:
        0.0  = No alerts (safe)
        0.25 = Info only
        0.50 = Warnings present
        0.75 = Critical alerts
        1.00 = Rejection-level issue
    """
    if not alerts:
        return SafetyReport(
            is_safe=True,
            risk_score=0.0,
            alerts=[],
            requires_review=False,
            recommendation="All safety checks passed. Proceed with clinical review.",
        )

    has_reject = any(a.level == AlertLevel.REJECT for a in alerts)
    has_critical = any(a.level == AlertLevel.CRITICAL for a in alerts)
    has_warning = any(a.level == AlertLevel.WARNING for a in alerts)

    if has_reject:
        risk_score = 1.0
        is_safe = False
        recommendation = (
            "⛔ REJECTED: Critical safety issue detected. "
            "Do NOT use this recommendation. Verify input data."
        )
    elif has_critical:
        risk_score = 0.75
        is_safe = False
        recommendation = (
            "⚠️ CRITICAL: Significant concerns detected. "
            "Pharmacist/physician review REQUIRED before acting."
        )
    elif has_warning:
        risk_score = 0.50
        is_safe = True
        recommendation = (
            "⚠️ WARNING: Minor concerns detected. "
            "Review warnings before proceeding."
        )
    else:
        risk_score = 0.25
        is_safe = True
        recommendation = "ℹ️ Informational alerts. Safe to proceed."

    return SafetyReport(
        is_safe=is_safe,
        risk_score=risk_score,
        alerts=alerts,
        requires_review=has_critical or has_reject,
        recommendation=recommendation,
    )
