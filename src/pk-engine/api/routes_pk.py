"""
PK Engine API Routes – Concentration prediction and clinical calculations.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import (
    PKPredictRequest, PKPredictResponse,
    ClinicalCalcRequest, ClinicalCalcResponse,
    SafetyReportSchema, SafetyAlertSchema,
)
from api.safety import (
    validate_patient, validate_doses, generate_safety_report,
    validate_pk_params,
)
from pk.models import (
    DoseEvent, Gender, ModelType, Observation, PatientData, Route,
)
from pk.population import get_model, compute_vancomycin_tv
from pk.solver import predict_concentrations
from pk.clinical import (
    cockcroft_gault_crcl, ckd_epi_egfr,
    ideal_body_weight, adjusted_body_weight,
    compute_weight_for_dosing,
)
from pk.analytical import auc_trapezoidal as compute_auc_trapezoidal


router = APIRouter(prefix="/pk", tags=["PK Engine"])


# ── Example request bodies for Swagger UI ────────────────────────

PREDICT_EXAMPLE = {
    "patient": {
        "age": 55, "weight": 70, "height": 170,
        "gender": "male", "serum_creatinine": 1.2,
    },
    "doses": [
        {"time": 0, "amount": 1000, "duration": 1.0, "route": "iv_infusion"},
        {"time": 12, "amount": 1000, "duration": 1.0, "route": "iv_infusion"},
    ],
    "times": [1, 2, 4, 6, 8, 10, 11.5, 13, 18, 23.5],
    "drug": "vancomycin_vn",
}


def _to_patient_data(p) -> PatientData:
    """Convert Pydantic schema to internal PatientData."""
    return PatientData(
        age=p.age,
        weight=p.weight,
        height=p.height,
        gender=Gender.MALE if p.gender.value == "male" else Gender.FEMALE,
        serum_creatinine=p.serum_creatinine,
        albumin=getattr(p, "albumin", 4.0),
        is_icu=getattr(p, "is_icu", False),
        is_on_dialysis=getattr(p, "is_on_dialysis", False),
    )


def _to_dose_events(doses) -> list[DoseEvent]:
    """Convert Pydantic schemas to internal DoseEvent list."""
    route_map = {
        "iv_bolus": Route.IV_BOLUS,
        "iv_infusion": Route.IV_INFUSION,
        "oral": Route.ORAL,
    }
    return [
        DoseEvent(
            time=d.time,
            amount=d.amount,
            duration=d.duration,
            route=route_map.get(d.route.value, Route.IV_INFUSION),
        )
        for d in doses
    ]


def _safety_to_schema(safety) -> SafetyReportSchema:
    """Convert Safety report to Pydantic schema."""
    return SafetyReportSchema(
        is_safe=safety.is_safe,
        risk_score=safety.risk_score,
        alerts=[SafetyAlertSchema(**a.__dict__) for a in safety.alerts],
        requires_review=safety.requires_review,
        recommendation=safety.recommendation,
    )


@router.post(
    "/predict",
    response_model=PKPredictResponse,
    summary="Predict drug concentrations over time",
)
def predict_concentrations_endpoint(req: PKPredictRequest) -> PKPredictResponse:
    """
    Predict drug concentration at specified time points.

    **Example patient**: Male, 55 tuổi, 70 kg, SCr 1.2 mg/dL.

    Uses covariate-adjusted typical values and ODE solver.
    Includes AUC₀₋₂₄ calculation and safety validation.
    """
    # Safety Layer 1: validate inputs
    alerts = []
    patient = _to_patient_data(req.patient)
    alerts.extend(validate_patient(patient))
    dose_events = _to_dose_events(req.doses)
    alerts.extend(validate_doses(dose_events))

    # Compute typical values (with error handling)
    try:
        tv = compute_vancomycin_tv(patient)
    except (ValueError, ZeroDivisionError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot compute PK parameters: {e}. "
                   f"Check patient data (weight, age, SCr must be positive).",
        )

    # Safety Layer 2: validate PK params
    alerts.extend(validate_pk_params(tv))

    # Predict concentrations
    try:
        model = get_model(req.drug)
        concs = predict_concentrations(
            tv, dose_events, req.times, model.model_type,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # AUC calculation
    auc = None
    if len(req.times) >= 2:
        try:
            auc = float(compute_auc_trapezoidal(req.times, concs.tolist()))
        except Exception:
            pass

    safety = generate_safety_report(alerts)

    return PKPredictResponse(
        times=req.times,
        concentrations=[round(float(c), 4) for c in concs],
        auc_0_24=round(auc, 2) if auc else None,
        safety=_safety_to_schema(safety),
    )


@router.post(
    "/clinical",
    response_model=ClinicalCalcResponse,
    summary="Compute clinical parameters (CrCL, eGFR, dosing weight)",
)
def clinical_calculations(req: ClinicalCalcRequest) -> ClinicalCalcResponse:
    """
    Compute derived clinical parameters for dosing decisions.

    Returns CrCL, eGFR, IBW, ABW, and dosing weight.
    """
    patient = _to_patient_data(req.patient)

    try:
        dosing_wt = compute_weight_for_dosing(patient)
        crcl = cockcroft_gault_crcl(
            age=patient.age,
            weight=dosing_wt,
            serum_creatinine=patient.serum_creatinine,
            gender=patient.gender,
        )
        egfr = ckd_epi_egfr(
            serum_creatinine=patient.serum_creatinine,
            age=patient.age,
            gender=patient.gender,
        )
        ibw = ideal_body_weight(
            height_cm=patient.height, gender=patient.gender,
        )
        abw = adjusted_body_weight(
            total_body_weight=patient.weight,
            height_cm=patient.height,
            gender=patient.gender,
        )
    except (ValueError, ZeroDivisionError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Clinical calculation error: {e}",
        )

    return ClinicalCalcResponse(
        crcl_ml_min=round(crcl, 2),
        egfr_ml_min=round(egfr, 2),
        abw_kg=round(abw, 2),
        ibw_kg=round(ibw, 2),
        dosing_weight_kg=round(dosing_wt, 2),
    )
