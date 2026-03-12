"""
Dosing Recommendation API Routes – Dose optimization with safety limits.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import (
    DosingRequest, DosingResponse,
    CFRRequest, CFRResponse,
    SafetyReportSchema, SafetyAlertSchema,
)
from api.safety import (
    validate_patient, validate_doses, validate_observations,
    validate_pk_params, validate_recommended_dose,
    generate_safety_report,
)
from api.routes_pk import _to_patient_data, _to_dose_events
from pk.models import Observation, ModelType
from pk.population import get_model, compute_vancomycin_tv
from pk.solver import predict_concentrations
from bayesian.map_estimator import estimate_map
from dosing.optimizer import optimize_dose, monte_carlo_pta, compute_cfr, PKPDTarget, DoseEvent
from pk.models import Route


router = APIRouter(prefix="/dosing", tags=["Dosing Recommendation"])


@router.post(
    "/recommend",
    response_model=DosingResponse,
    summary="Get optimized dose recommendation with safety validation",
)
def recommend_dose(req: DosingRequest) -> DosingResponse:
    """
    Compute optimal vancomycin dose for AUC₂₄/MIC target.

    Pipeline:
        1. Validate all inputs (Layer 1)
        2. Estimate individual PK via MAP (if TDM available)
        3. Optimize dose for AUC target
        4. Validate recommendation (Layer 3)
        5. Compute PTA (probability of target attainment)
    """
    alerts = []
    patient = _to_patient_data(req.patient)
    alerts.extend(validate_patient(patient))
    dose_events = _to_dose_events(req.doses)
    alerts.extend(validate_doses(dose_events))
    if req.observations:
        alerts.extend(validate_observations(
            [{"concentration": o.concentration, "time": o.time}
             for o in req.observations]
        ))

    model = get_model(req.drug)
    try:
        tv = compute_vancomycin_tv(patient)
    except (ValueError, ZeroDivisionError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot compute PK parameters: {e}. "
                   f"Check patient data (weight, age, SCr must be positive).",
        )

    # If TDM observations provided, use MAP estimation
    if req.observations:
        observations = [
            Observation(
                time=o.time,
                concentration=o.concentration,
                sample_type=o.sample_type,
            )
            for o in req.observations
        ]
        try:
            map_result = estimate_map(
                model, tv, dose_events, observations,
            )
            ind_params = map_result.params
        except (ValueError, RuntimeError):
            ind_params = tv  # Fallback to population values
    else:
        ind_params = tv

    # Validate estimated PK params
    alerts.extend(validate_pk_params(ind_params))

    # Optimize dose using PKPDTarget
    target = PKPDTarget(
        auc24_min=req.target_auc_min,
        auc24_max=req.target_auc_max,
    )

    try:
        opt_result = optimize_dose(
            params=ind_params,
            model_type=model.model_type,
            target=target,
        )
        rec_dose = opt_result.dose_mg
        rec_interval = opt_result.interval_h
        pred_auc = opt_result.predicted_auc24
        pred_trough = opt_result.predicted_trough
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate recommended dose (Layer 3)
    alerts.extend(validate_recommended_dose(
        rec_dose,
        current_dose=req.current_dose,
    ))

    # PTA via Monte Carlo
    try:
        pta_result = monte_carlo_pta(
            tv_params=ind_params,
            model=model,
            dose_mg=rec_dose,
            interval_h=rec_interval,
            target=target,
            n_simulations=1000,  # Fewer for API speed
        )
        pta = pta_result
    except Exception:
        pta = 0.0

    safety = generate_safety_report(alerts)

    return DosingResponse(
        recommended_dose_mg=round(rec_dose, 0),
        recommended_interval_h=round(rec_interval, 1),
        predicted_auc=round(pred_auc, 1),
        predicted_trough=round(pred_trough, 2),
        pta_probability=round(pta * 100, 1),
        safety=SafetyReportSchema(**{
            "is_safe": safety.is_safe,
            "risk_score": safety.risk_score,
            "alerts": [SafetyAlertSchema(**a.__dict__) for a in safety.alerts],
            "requires_review": safety.requires_review,
            "recommendation": safety.recommendation,
        }),
    )


@router.post(
    "/cfr",
    response_model=CFRResponse,
    summary="Compute Cumulative Fraction of Response (CFR)",
)
def dosing_cfr(req: CFRRequest) -> CFRResponse:
    """
    Cumulative Fraction of Response (CFR).

    CFR = Σ PTA(MIC) × F(MIC)

    Accounts for the MIC distribution of the pathogen population
    to compute the overall probability that a regimen will be effective.
    """
    alerts = []
    patient = _to_patient_data(req.patient)
    alerts.extend(validate_patient(patient))

    model = get_model(req.drug)
    try:
        tv = compute_vancomycin_tv(patient)
    except (ValueError, ZeroDivisionError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot compute PK parameters: {e}.",
        )

    try:
        cfr_result = compute_cfr(
            tv_params=tv,
            model=model,
            dose_mg=req.dose_mg,
            interval_h=req.interval_h,
            mic_distribution=req.mic_distribution,
            n_simulations=req.n_simulations,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    safety = generate_safety_report(alerts)

    return CFRResponse(
        cfr=round(cfr_result.cfr, 4),
        cfr_percent=round(cfr_result.cfr * 100, 1),
        pta_by_mic={str(k): round(v, 4) for k, v in cfr_result.pta_by_mic.items()},
        mic_distribution={str(k): v for k, v in cfr_result.mic_distribution.items()},
        dose_mg=cfr_result.dose_mg,
        interval_h=cfr_result.interval_h,
        safety=SafetyReportSchema(**{
            "is_safe": safety.is_safe,
            "risk_score": safety.risk_score,
            "alerts": [SafetyAlertSchema(**a.__dict__) for a in safety.alerts],
            "requires_review": safety.requires_review,
            "recommendation": safety.recommendation,
        }),
    )
