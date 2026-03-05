"""
SHAP Explainer – Feature importance explanation for PK predictions.

Implements Shapley value approximation for explaining
why a patient's PK parameters differ from population.

phi_i = sum over S subset of N without i:
    [|S|! * (M-|S|-1)!] / M!  *  [f(S union i) - f(S)]

Efficiency: sum(phi_i) = f(x) - E[f(x)]

Uses a simplified kernel SHAP approach for PopPK covariate models.

Reference: Lundberg & Lee (2017), NeurIPS
Dependencies: numpy, pk.models
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from pk.models import Gender, PatientData


# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────

@dataclass
class ShapFeature:
    """SHAP value for one feature."""
    name: str
    value: float         # Actual feature value
    shap_value: float    # SHAP contribution
    direction: str       # "increases" or "decreases" parameter


@dataclass
class ShapExplanation:
    """Complete SHAP explanation for a PK parameter."""
    parameter_name: str           # e.g., "CL"
    base_value: float             # E[f(x)] = population typical value
    predicted_value: float        # f(x) = individual value
    features: list[ShapFeature]   # Sorted by |SHAP| descending


# ──────────────────────────────────────────────────────────────────
# Reference (median) population values for centering
# ──────────────────────────────────────────────────────────────────

_REFERENCE_VALUES = {
    "age": 55.0,
    "weight": 70.0,
    "height": 170.0,
    "serum_creatinine": 1.0,
    "crcl": 85.0,
    "albumin": 3.5,
    "is_icu": 0.0,
    "is_on_dialysis": 0.0,
}


# ──────────────────────────────────────────────────────────────────
# Marginal contribution estimation
# ──────────────────────────────────────────────────────────────────

def _compute_covariate_effect(
    feature_name: str,
    feature_value: float,
    reference_value: float,
    pk_param: str,
) -> float:
    """
    Estimate marginal effect of one covariate on PK parameter.

    Based on the vancomycin covariate model:
        CL: (CrCL/85)^0.75 * (WT/70)^0.75
        V1: (WT/70)^1.0

    Returns the log-ratio contribution (multiplicative effect).
    """
    if reference_value <= 0:
        return 0.0

    ratio = feature_value / reference_value

    if ratio <= 0:
        return 0.0

    # Covariate exponents for CL
    cl_exponents = {
        "crcl": 0.75,
        "weight": 0.75,
        "albumin": 0.0,    # Placeholder
        "age": 0.0,        # Implicit via CrCL
        "is_icu": 0.0,
    }

    # Covariate exponents for V1
    v1_exponents = {
        "weight": 1.0,
        "crcl": 0.0,
        "albumin": 0.0,
        "age": 0.0,
        "is_icu": 0.0,
    }

    if pk_param == "CL":
        exponent = cl_exponents.get(feature_name, 0.0)
    elif pk_param == "V1":
        exponent = v1_exponents.get(feature_name, 0.0)
    else:
        exponent = 0.0

    if exponent == 0.0:
        return 0.0

    # Multiplicative contribution: ratio^exponent
    # SHAP value (on log scale): exponent * ln(ratio)
    return exponent * np.log(ratio)


# ──────────────────────────────────────────────────────────────────
# Main SHAP explanation
# ──────────────────────────────────────────────────────────────────

def explain_pk_parameter(
    patient: PatientData,
    crcl: float,
    pk_param: str,
    base_value: float,
    predicted_value: float,
) -> ShapExplanation:
    """
    Generate SHAP-like explanation for a PK parameter prediction.

    Computes marginal contribution of each covariate to the
    difference between population typical and individual value.

    Args:
        patient:         Patient data
        crcl:            Computed CrCL (mL/min)
        pk_param:        Parameter name ("CL" or "V1")
        base_value:      Population typical value
        predicted_value: Individual predicted value

    Returns:
        ShapExplanation with feature contributions
    """
    features_dict = {
        "crcl": crcl,
        "weight": patient.weight,
        "age": patient.age,
        "albumin": patient.albumin if patient.albumin is not None else 3.5,
        "is_icu": 1.0 if patient.is_icu else 0.0,
    }

    shap_features: list[ShapFeature] = []

    for feat_name, feat_value in features_dict.items():
        ref_value = _REFERENCE_VALUES.get(feat_name, feat_value)
        log_contribution = _compute_covariate_effect(
            feat_name, feat_value, ref_value, pk_param
        )

        # Convert from log contribution to absolute contribution
        abs_contribution = base_value * (np.exp(log_contribution) - 1.0)

        if abs(abs_contribution) < 0.001:
            direction = "không ảnh hưởng"
        elif abs_contribution > 0:
            direction = f"tăng {pk_param}"
        else:
            direction = f"giảm {pk_param}"

        shap_features.append(ShapFeature(
            name=feat_name,
            value=feat_value,
            shap_value=float(abs_contribution),
            direction=direction,
        ))

    # Sort by absolute SHAP value (descending)
    shap_features.sort(key=lambda f: abs(f.shap_value), reverse=True)

    return ShapExplanation(
        parameter_name=pk_param,
        base_value=base_value,
        predicted_value=predicted_value,
        features=shap_features,
    )
