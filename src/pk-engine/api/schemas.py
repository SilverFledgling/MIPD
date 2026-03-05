"""
Pydantic Schemas – Request/Response models for PK Engine API.

All API endpoints use these schemas for:
    - Input validation (Pydantic enforces types + constraints)
    - OpenAPI documentation (auto-generated from models)
    - JSON serialization/deserialization
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────

class GenderEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"


class RouteEnum(str, Enum):
    IV_BOLUS = "iv_bolus"
    IV_INFUSION = "iv_infusion"
    ORAL = "oral"


class ModelTypeEnum(str, Enum):
    ONE_COMP_IV = "one_comp_iv"
    TWO_COMP_IV = "two_comp_iv"
    ONE_COMP_ORAL = "one_comp_oral"
    TWO_COMP_ORAL = "two_comp_oral"


class InferenceMethodEnum(str, Enum):
    MAP = "map"
    LAPLACE = "laplace"
    ADVI = "advi"
    EP = "ep"
    SMC = "smc"


# ──────────────────────────────────────────────────────────────────
# Patient
# ──────────────────────────────────────────────────────────────────

class PatientSchema(BaseModel):
    """Patient demographics."""
    age: float = Field(..., ge=0, le=120, description="Age in years")
    weight: float = Field(..., ge=10, le=300, description="Weight in kg")
    height: float = Field(..., ge=50, le=250, description="Height in cm")
    gender: GenderEnum = Field(..., description="Patient gender")
    serum_creatinine: float = Field(
        ..., ge=0.1, le=20.0, description="Serum creatinine (mg/dL)",
    )
    albumin: float = Field(default=4.0, ge=0.5, le=6.0, description="Albumin (g/dL)")
    is_icu: bool = Field(default=False, description="ICU admission status")
    is_on_dialysis: bool = Field(default=False, description="Dialysis status")


# ──────────────────────────────────────────────────────────────────
# Dose and Observation
# ──────────────────────────────────────────────────────────────────

class DoseSchema(BaseModel):
    """Drug administration event."""
    time: float = Field(..., ge=0, description="Time of dose (hours)")
    amount: float = Field(..., gt=0, le=10000, description="Dose amount (mg)")
    duration: float = Field(default=1.0, ge=0, description="Infusion duration (hours)")
    route: RouteEnum = Field(default=RouteEnum.IV_INFUSION)


class ObservationSchema(BaseModel):
    """TDM concentration measurement."""
    time: float = Field(..., ge=0, description="Sampling time (hours)")
    concentration: float = Field(
        ..., ge=0, le=200, description="Measured concentration (mg/L)",
    )
    sample_type: str = Field(default="trough", description="Sample type")


# ──────────────────────────────────────────────────────────────────
# Safety
# ──────────────────────────────────────────────────────────────────

class SafetyAlertSchema(BaseModel):
    """Safety alert."""
    level: str
    code: str
    message: str
    field: str = ""
    value: Any = ""


class SafetyReportSchema(BaseModel):
    """Safety assessment report."""
    is_safe: bool
    risk_score: float
    alerts: list[SafetyAlertSchema] = []
    requires_review: bool = False
    recommendation: str = ""


# ──────────────────────────────────────────────────────────────────
# PK Endpoints
# ──────────────────────────────────────────────────────────────────

class PKPredictRequest(BaseModel):
    """Request: predict concentrations."""
    patient: PatientSchema
    doses: list[DoseSchema]
    times: list[float] = Field(..., min_length=1, description="Prediction times (h)")
    drug: str = Field(default="vancomycin_vn")

    model_config = {"json_schema_extra": {"examples": [{
        "patient": {"age": 55, "weight": 70, "height": 170,
                    "gender": "male", "serum_creatinine": 1.2},
        "doses": [{"time": 0, "amount": 1000, "duration": 1.0,
                   "route": "iv_infusion"}],
        "times": [1, 4, 8, 11.5],
    }]}}


class PKPredictResponse(BaseModel):
    """Response: predicted concentration profile."""
    times: list[float]
    concentrations: list[float]
    auc_0_24: float | None = None
    safety: SafetyReportSchema


class ClinicalCalcRequest(BaseModel):
    """Request: clinical calculations."""
    patient: PatientSchema


class ClinicalCalcResponse(BaseModel):
    """Response: clinical calculations."""
    crcl_ml_min: float
    egfr_ml_min: float
    abw_kg: float
    ibw_kg: float
    dosing_weight_kg: float


# ──────────────────────────────────────────────────────────────────
# Bayesian Endpoints
# ──────────────────────────────────────────────────────────────────

class BayesianRequest(BaseModel):
    """Request: Bayesian PK parameter estimation."""
    patient: PatientSchema
    doses: list[DoseSchema]
    observations: list[ObservationSchema] = Field(..., min_length=1)
    drug: str = Field(default="vancomycin_vn")
    method: InferenceMethodEnum = Field(default=InferenceMethodEnum.MAP)

    model_config = {"json_schema_extra": {"examples": [{
        "patient": {"age": 55, "weight": 70, "height": 170,
                    "gender": "male", "serum_creatinine": 1.2},
        "doses": [{"time": 0, "amount": 1000, "duration": 1.0,
                   "route": "iv_infusion"},
                  {"time": 12, "amount": 1000, "duration": 1.0,
                   "route": "iv_infusion"}],
        "observations": [{"time": 11.5, "concentration": 12.0,
                          "sample_type": "trough"}],
        "method": "map",
    }]}}


class PKParamsSchema(BaseModel):
    """Estimated PK parameters."""
    CL: float = Field(description="Clearance (L/h)")
    V1: float = Field(description="Volume of central compartment (L)")
    Q: float = Field(default=0, description="Inter-compartmental clearance (L/h)")
    V2: float = Field(default=0, description="Peripheral volume (L)")


class BayesianResponse(BaseModel):
    """Response: Bayesian estimation result."""
    method: str
    individual_params: PKParamsSchema
    eta: list[float] = Field(description="Random effects")
    confidence: dict[str, dict[str, float]] | None = None
    diagnostics: dict[str, Any] = {}
    safety: SafetyReportSchema


# ──────────────────────────────────────────────────────────────────
# Dosing Endpoints
# ──────────────────────────────────────────────────────────────────

class DosingRequest(BaseModel):
    """Request: dose optimization."""
    patient: PatientSchema
    doses: list[DoseSchema]
    observations: list[ObservationSchema] = []
    drug: str = Field(default="vancomycin_vn")
    target_auc_min: float = Field(default=400, description="AUC target min (mg*h/L)")
    target_auc_max: float = Field(default=600, description="AUC target max (mg*h/L)")
    current_dose: float | None = Field(default=None, description="Current dose (mg)")

    model_config = {"json_schema_extra": {"examples": [{
        "patient": {"age": 55, "weight": 70, "height": 170,
                    "gender": "male", "serum_creatinine": 1.2},
        "doses": [{"time": 0, "amount": 1000, "duration": 1.0,
                   "route": "iv_infusion"}],
        "observations": [{"time": 11.5, "concentration": 12.0,
                          "sample_type": "trough"}],
        "current_dose": 1000,
    }]}}


class DosingResponse(BaseModel):
    """Response: dose recommendation."""
    recommended_dose_mg: float
    recommended_interval_h: float
    predicted_auc: float
    predicted_trough: float
    pta_probability: float = Field(description="Prob. of target attainment (%)")
    safety: SafetyReportSchema


# ──────────────────────────────────────────────────────────────────
# AI/ML Endpoints
# ──────────────────────────────────────────────────────────────────

class AnomalyCheckRequest(BaseModel):
    """Request: TDM anomaly detection."""
    patient: PatientSchema
    doses: list[DoseSchema]
    observations: list[ObservationSchema] = Field(..., min_length=1)
    drug: str = Field(default="vancomycin_vn")


class AnomalyCheckResponse(BaseModel):
    """Response: anomaly detection result."""
    quality_score: float = Field(description="0-1, higher = better quality")
    verdict: str = Field(description="ACCEPT / WARNING / REJECT")
    head_scores: dict[str, float]
    alerts: list[SafetyAlertSchema] = []


class CovariateScreenRequest(BaseModel):
    """Request: covariate screening."""
    X: list[list[float]]
    y: list[float]
    feature_names: list[str]
    top_k: int = Field(default=3, ge=1, le=20)


class CovariateScreenResponse(BaseModel):
    """Response: covariate screening result."""
    ranking: list[str]
    selected: list[str]
    rf_importance: list[float]
    nn_importance: list[float]
    svr_importance: list[float]
    borda_scores: list[float]


# ──────────────────────────────────────────────────────────────────
# Validation Endpoints
# ──────────────────────────────────────────────────────────────────

class ValidationMetricsRequest(BaseModel):
    """Request: compute validation metrics."""
    observed: list[float]
    predicted: list[float]


class ValidationMetricsResponse(BaseModel):
    """Response: validation metrics."""
    mpe: float
    mape: float
    rmse: float
    ccc: float
    bland_altman: dict[str, float]
    safety: SafetyReportSchema
