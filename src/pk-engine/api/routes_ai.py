"""
AI/ML and Validation API Routes – Anomaly detection, screening, metrics.
"""

from fastapi import APIRouter, HTTPException
import numpy as np

from api.schemas import (
    AnomalyCheckRequest, AnomalyCheckResponse,
    CovariateScreenRequest, CovariateScreenResponse,
    ValidationMetricsRequest, ValidationMetricsResponse,
    SafetyReportSchema, SafetyAlertSchema,
)
from api.safety import generate_safety_report, SafetyAlert, AlertLevel
from api.routes_pk import _to_patient_data, _to_dose_events
from ai.anomaly_detection import detect_anomaly, AnomalyResult
from ai.ml_screening import screen_covariates
from validation.metrics import compute_metrics
from pk.population import compute_vancomycin_tv
from pk.solver import predict_concentrations
from pk.models import ModelType

router = APIRouter(prefix="/ai", tags=["AI/ML & Validation"])


@router.post(
    "/anomaly-check",
    response_model=AnomalyCheckResponse,
    summary="Check TDM sample quality using Swift Hydra 4-head detector",
)
def anomaly_check(req: AnomalyCheckRequest) -> AnomalyCheckResponse:
    """
    Run Swift Hydra anomaly detection on TDM observations.

    4 heads: Range check, Time-series deviation, Dose-response plausibility,
    Isolation Forest. Returns quality score and ACCEPT/WARNING/REJECT verdict.
    """
    patient = _to_patient_data(req.patient)
    dose_events = _to_dose_events(req.doses)

    tv = compute_vancomycin_tv(patient)

    # Run anomaly detection per observation
    results = []
    for obs in req.observations:
        # Predict expected concentration
        try:
            c_pred_arr = predict_concentrations(
                tv, dose_events, [obs.time], ModelType.TWO_COMP_IV,
            )
            c_pred = float(c_pred_arr[0])
        except Exception:
            c_pred = 15.0  # Default

        result = detect_anomaly(
            c_obs=obs.concentration,
            c_predicted=c_pred,
            population_mean=15.0,   # Typical trough ~15 mg/L
            population_sd=5.0,      # Typical SD
            residual_sd=2.0,        # Typical residual
            omega_cl=0.25,          # Typical omega CL
        )
        results.append(result)

    # Aggregate: use worst quality score
    worst = min(results, key=lambda r: r.quality_score)
    head_scores = {
        h.name: round(h.score, 3)
        for h in worst.heads
    }

    # Generate alerts
    alerts: list[SafetyAlertSchema] = []
    if worst.verdict.value == "REJECT":
        alerts.append(SafetyAlertSchema(
            level="critical",
            code="TDM_ANOMALY_DETECTED",
            message=f"TDM data quality REJECTED (score={worst.quality_score:.2f})",
        ))
    elif worst.verdict.value == "WARNING":
        alerts.append(SafetyAlertSchema(
            level="warning",
            code="TDM_QUALITY_WARNING",
            message=f"TDM data quality WARNING (score={worst.quality_score:.2f})",
        ))

    return AnomalyCheckResponse(
        quality_score=round(worst.quality_score, 3),
        verdict=worst.verdict.value.upper(),
        head_scores=head_scores,
        alerts=alerts,
    )


@router.post(
    "/screen-covariates",
    response_model=CovariateScreenResponse,
    summary="Screen covariates using RF + NN + SVR with Borda ranking",
)
def screen_covariates_endpoint(
    req: CovariateScreenRequest,
) -> CovariateScreenResponse:
    """
    Identify most important covariates for PK parameter prediction.

    Uses RF, NN, and SVR with permutation importance,
    aggregated via Borda count.
    """
    X = np.array(req.X, dtype=np.float64)
    y = np.array(req.y, dtype=np.float64)

    if X.shape[0] != len(y):
        raise HTTPException(
            status_code=400, detail="X rows must match y length",
        )
    if X.shape[1] != len(req.feature_names):
        raise HTTPException(
            status_code=400, detail="X columns must match feature_names",
        )

    try:
        result = screen_covariates(
            X, y, req.feature_names, top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return CovariateScreenResponse(
        ranking=result.ranking,
        selected=result.selected,
        rf_importance=result.rf_importance.tolist(),
        nn_importance=result.nn_importance.tolist(),
        svr_importance=result.svr_importance.tolist(),
        borda_scores=result.borda_scores.tolist(),
    )


@router.post(
    "/validate-metrics",
    response_model=ValidationMetricsResponse,
    summary="Compute prediction validation metrics (MPE, MAPE, RMSE, CCC)",
)
def validate_metrics(req: ValidationMetricsRequest) -> ValidationMetricsResponse:
    """
    Compute statistical validation metrics for model predictions.

    Returns MPE, MAPE, RMSE, CCC, and Bland-Altman analysis.
    """
    obs = np.array(req.observed, dtype=np.float64)
    pred = np.array(req.predicted, dtype=np.float64)

    if len(obs) != len(pred):
        raise HTTPException(
            status_code=400, detail="observed and predicted must have same length",
        )
    if len(obs) < 2:
        raise HTTPException(
            status_code=400, detail="Need at least 2 data points",
        )

    # compute_metrics(estimated, true_values)
    metrics = compute_metrics(pred, obs)

    # Safety: flag poor model performance
    alerts: list[SafetyAlert] = []
    if abs(metrics.mpe) > 20:
        alerts.append(SafetyAlert(
            level=AlertLevel.WARNING,
            code="HIGH_BIAS",
            message=f"Model bias (MPE={metrics.mpe:.1f}%) exceeds 20%",
            field="mpe",
            value=round(metrics.mpe, 2),
        ))
    if metrics.mape > 30:
        alerts.append(SafetyAlert(
            level=AlertLevel.CRITICAL,
            code="HIGH_IMPRECISION",
            message=f"Model imprecision (MAPE={metrics.mape:.1f}%) exceeds 30%",
            field="mape",
            value=round(metrics.mape, 2),
        ))

    safety = generate_safety_report(alerts)

    return ValidationMetricsResponse(
        mpe=round(metrics.mpe, 4),
        mape=round(metrics.mape, 4),
        rmse=round(metrics.rmse, 4),
        ccc=round(metrics.ccc, 4),
        bland_altman={
            "mean_diff": round(metrics.bias, 4),
            "upper_loa": round(metrics.loa_upper, 4),
            "lower_loa": round(metrics.loa_lower, 4),
        },
        safety=SafetyReportSchema(**{
            "is_safe": safety.is_safe,
            "risk_score": safety.risk_score,
            "alerts": [SafetyAlertSchema(**a.__dict__) for a in safety.alerts],
            "requires_review": safety.requires_review,
            "recommendation": safety.recommendation,
        }),
    )
