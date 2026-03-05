"""
Swift Hydra Anomaly Detection – Multi-head data quality checker for TDM samples.

4 detection heads:
    Head 1 (Range):        z = (C_obs - mu_exp) / sigma_exp, flag if |z| > 3
    Head 2 (Time-series):  delta = |C_obs - C_pred| / sigma_exp, flag if > 2.5
    Head 3 (Dose-Response): C in [f(dose,theta)*exp(-2*omega), f(dose,theta)*exp(+2*omega)]
    Head 4 (Isolation Forest): anomaly score, flag if > 0.7

Quality = 1 - max(scores)
    >= 0.8 ACCEPT | 0.5-0.8 WARNING | < 0.5 REJECT

Dependencies: numpy, pk.models
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from enum import Enum

from pk.models import ErrorModel, PKParams


# ──────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────

class QualityVerdict(str, Enum):
    """TDM sample quality verdict."""
    ACCEPT = "ACCEPT"
    WARNING = "WARNING"
    REJECT = "REJECT"


@dataclass
class HeadResult:
    """Result from one detection head."""
    name: str
    score: float       # 0 = no anomaly, 1 = maximum anomaly
    flagged: bool
    detail: str


@dataclass
class AnomalyResult:
    """Complete anomaly detection result."""
    quality_score: float        # 1 - max(head scores)
    verdict: QualityVerdict
    heads: list[HeadResult]
    is_valid: bool


# ──────────────────────────────────────────────────────────────────
# Head 1: Range Check (z-score)
# ──────────────────────────────────────────────────────────────────

def _head_range_check(
    c_obs: float,
    population_mean: float,
    population_sd: float,
    threshold: float = 3.0,
) -> HeadResult:
    """
    Range check: flag if concentration is > threshold SDs from mean.

    z = (C_obs - mu) / sigma
    """
    if population_sd <= 0:
        return HeadResult(
            name="Range Check",
            score=0.0,
            flagged=False,
            detail="SD is zero, skip",
        )

    z = abs(c_obs - population_mean) / population_sd
    score = min(z / (threshold + 1.0), 1.0)
    flagged = z > threshold

    return HeadResult(
        name="Range Check",
        score=score,
        flagged=flagged,
        detail=f"z={z:.2f} ({'FLAG' if flagged else 'OK'}, threshold={threshold})",
    )


# ──────────────────────────────────────────────────────────────────
# Head 2: Time-Series Deviation
# ──────────────────────────────────────────────────────────────────

def _head_time_series(
    c_obs: float,
    c_predicted: float,
    residual_sd: float,
    threshold: float = 2.5,
) -> HeadResult:
    """
    Time-series check: deviation from model prediction.

    delta = |C_obs - C_pred| / sigma_residual
    """
    if residual_sd <= 0:
        residual_sd = 1.0

    delta = abs(c_obs - c_predicted) / residual_sd
    score = min(delta / (threshold + 0.5), 1.0)
    flagged = delta > threshold

    return HeadResult(
        name="Time-Series Deviation",
        score=score,
        flagged=flagged,
        detail=f"delta={delta:.2f} ({'FLAG' if flagged else 'OK'}, threshold={threshold})",
    )


# ──────────────────────────────────────────────────────────────────
# Head 3: Dose-Response Plausibility
# ──────────────────────────────────────────────────────────────────

def _head_dose_response(
    c_obs: float,
    c_model: float,
    omega_cl: float,
) -> HeadResult:
    """
    Dose-response plausibility: check if C_obs falls within
    2*omega range of model prediction (accounting for IIV).

    Expected range: [C_model * exp(-2*omega), C_model * exp(+2*omega)]
    """
    if c_model <= 0 or omega_cl <= 0:
        return HeadResult(
            name="Dose-Response",
            score=0.0,
            flagged=False,
            detail="Insufficient model data, skip",
        )

    lower_bound = c_model * np.exp(-2.0 * omega_cl)
    upper_bound = c_model * np.exp(2.0 * omega_cl)

    in_range = lower_bound <= c_obs <= upper_bound

    if in_range:
        score = 0.0
    else:
        # How far outside the range
        if c_obs < lower_bound:
            deviation = (lower_bound - c_obs) / lower_bound
        else:
            deviation = (c_obs - upper_bound) / upper_bound
        score = min(deviation, 1.0)

    return HeadResult(
        name="Dose-Response",
        score=score,
        flagged=not in_range,
        detail=f"C_obs={c_obs:.1f}, range=[{lower_bound:.1f}, {upper_bound:.1f}]",
    )


# ──────────────────────────────────────────────────────────────────
# Head 4: Isolation Forest (simplified score)
# ──────────────────────────────────────────────────────────────────

def _head_isolation_score(
    c_obs: float,
    historical_concentrations: NDArray[np.float64],
    threshold: float = 0.7,
) -> HeadResult:
    """
    Simplified isolation-forest-style anomaly score.

    Uses z-score approach on historical data as proxy.
    In full production: use sklearn.ensemble.IsolationForest.

    score = 2^(-E[h(x)] / c(n)) approximated via z-score mapping.
    """
    if len(historical_concentrations) < 3:
        return HeadResult(
            name="Isolation Forest",
            score=0.0,
            flagged=False,
            detail="Not enough historical data, skip",
        )

    mu = float(np.mean(historical_concentrations))
    sigma = float(np.std(historical_concentrations, ddof=1))

    if sigma <= 0:
        sigma = 1.0

    z = abs(c_obs - mu) / sigma
    # Map z-score to anomaly score [0, 1]
    score = float(1.0 - np.exp(-0.5 * (z / 2.0) ** 2))
    flagged = score > threshold

    return HeadResult(
        name="Isolation Forest",
        score=score,
        flagged=flagged,
        detail=f"score={score:.3f} ({'FLAG' if flagged else 'OK'}, threshold={threshold})",
    )


# ──────────────────────────────────────────────────────────────────
# Main detection function
# ──────────────────────────────────────────────────────────────────

def detect_anomaly(
    c_obs: float,
    c_predicted: float,
    population_mean: float,
    population_sd: float,
    residual_sd: float,
    omega_cl: float,
    historical_concentrations: NDArray[np.float64] | None = None,
) -> AnomalyResult:
    """
    Run all 4 Swift Hydra detection heads on a TDM sample.

    Args:
        c_obs:           Observed concentration (mg/L)
        c_predicted:     Model-predicted concentration (mg/L)
        population_mean: Population mean concentration at this time
        population_sd:   Population SD of concentration
        residual_sd:     Residual error SD from error model
        omega_cl:        IIV omega for CL (for dose-response range)
        historical_concentrations: Previous TDM values (optional)

    Returns:
        AnomalyResult with quality score and verdict
    """
    heads: list[HeadResult] = []

    # Head 1: Range check
    heads.append(_head_range_check(c_obs, population_mean, population_sd))

    # Head 2: Time-series deviation
    heads.append(_head_time_series(c_obs, c_predicted, residual_sd))

    # Head 3: Dose-response plausibility
    heads.append(_head_dose_response(c_obs, c_predicted, omega_cl))

    # Head 4: Isolation forest
    if historical_concentrations is not None:
        heads.append(_head_isolation_score(c_obs, historical_concentrations))

    # Overall quality score
    max_score = max(h.score for h in heads) if heads else 0.0
    quality = 1.0 - max_score

    # Verdict
    if quality >= 0.8:
        verdict = QualityVerdict.ACCEPT
    elif quality >= 0.5:
        verdict = QualityVerdict.WARNING
    else:
        verdict = QualityVerdict.REJECT

    return AnomalyResult(
        quality_score=quality,
        verdict=verdict,
        heads=heads,
        is_valid=quality >= 0.5,
    )
