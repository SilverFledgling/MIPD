"""
Dose Optimizer – Find optimal dosing regimen based on PK/PD targets.

Implements:
    1. Grid search over standard dose/interval combinations
    2. Monte Carlo Probability of Target Attainment (PTA)
    3. Cumulative Fraction of Response (CFR)

PK/PD Targets (Vancomycin – Rybak 2020):
    AUC24/MIC = 400-600 mg*h/L  (MIC ~ 1 mg/L)
    Safety: AUC24 < 700-800, C_trough < 20 mg/L

Reference: Rybak et al. (2020), AJHP, 77(11), 835-864
Dependencies: numpy, pk.models, pk.solver, pk.population
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from pk.models import DoseEvent, ModelType, PKParams, PopPKModel, Route
from pk.solver import simulate
from pk.population import sample_individual_params


# ──────────────────────────────────────────────────────────────────
# Standard dose options
# ──────────────────────────────────────────────────────────────────

VANCOMYCIN_DOSES_MG = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000]
VANCOMYCIN_INTERVALS_H = [6, 8, 12, 18, 24, 36, 48]
DEFAULT_INFUSION_DURATION_H = 1.0


# ──────────────────────────────────────────────────────────────────
# PK/PD targets
# ──────────────────────────────────────────────────────────────────

@dataclass
class PKPDTarget:
    """
    PK/PD target for dose optimization.

    Attributes:
        auc24_min:      Minimum AUC24/MIC (mg*h/L)
        auc24_max:      Maximum AUC24/MIC (mg*h/L)
        trough_max:     Maximum trough concentration (mg/L)
        mic:            MIC of pathogen (mg/L)
    """
    auc24_min: float = 400.0
    auc24_max: float = 600.0
    trough_max: float = 20.0
    mic: float = 1.0


# ──────────────────────────────────────────────────────────────────
# Optimization result
# ──────────────────────────────────────────────────────────────────

@dataclass
class DosingResult:
    """
    Result of dose optimization.

    Attributes:
        dose_mg:        Optimal dose (mg)
        interval_h:     Optimal dosing interval (hours)
        predicted_auc24: Predicted AUC24 (mg*h/L)
        predicted_auc24_mic: AUC24/MIC ratio
        predicted_trough: Predicted trough concentration (mg/L)
        predicted_peak:  Predicted peak concentration after infusion (mg/L)
        pta:            Probability of target attainment (0-1)
        infusion_h:     Infusion duration (hours)
        alternatives:   List of alternative regimens ranked by closeness
    """
    dose_mg: float
    interval_h: float
    predicted_auc24: float
    predicted_auc24_mic: float
    predicted_trough: float
    predicted_peak: float
    pta: float
    infusion_h: float = DEFAULT_INFUSION_DURATION_H
    alternatives: list[dict[str, float]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────
# AUC24 at steady state from simulation
# ──────────────────────────────────────────────────────────────────

def _compute_ss_auc24(
    params: PKParams,
    dose_mg: float,
    interval_h: float,
    model_type: ModelType,
    infusion_h: float = DEFAULT_INFUSION_DURATION_H,
    n_doses_to_ss: int = 10,
) -> tuple[float, float, float]:
    """
    Compute AUC24, trough, and peak at approximate steady state.

    Gives enough doses to reach steady state, then measures the
    last dosing interval.

    Args:
        params:         Individual PK parameters
        dose_mg:        Dose per administration (mg)
        interval_h:     Dosing interval (hours)
        model_type:     PK model type
        infusion_h:     Infusion duration (hours)
        n_doses_to_ss:  Number of doses for SS approximation

    Returns:
        Tuple of (AUC24, trough, peak)
    """
    # Build dose schedule
    route = Route.IV_INFUSION if infusion_h > 0 else Route.IV_BOLUS
    doses = [
        DoseEvent(
            time=i * interval_h,
            amount=dose_mg,
            duration=infusion_h,
            route=route,
        )
        for i in range(n_doses_to_ss)
    ]

    # Simulate
    t_end = n_doses_to_ss * interval_h
    result = simulate(params, doses, model_type, t_end=t_end, dt=0.05)

    # Last dosing interval for SS measurement
    t_ss_start = (n_doses_to_ss - 1) * interval_h
    t_ss_end = t_end

    # AUC in last interval, then scale to 24h
    auc_interval = result.auc_interval(t_ss_start, t_ss_end)
    auc24 = auc_interval * (24.0 / interval_h)

    # Trough: concentration at end of last interval
    trough = result.concentration_at(t_ss_end - 0.01)

    # Peak: concentration right after infusion ends
    t_peak = t_ss_start + infusion_h
    peak = result.concentration_at(t_peak)

    return auc24, trough, peak


# ──────────────────────────────────────────────────────────────────
# Grid search optimization
# ──────────────────────────────────────────────────────────────────

def optimize_dose(
    params: PKParams,
    model_type: ModelType,
    target: PKPDTarget | None = None,
    doses_mg: list[int] | None = None,
    intervals_h: list[int] | None = None,
    infusion_h: float = DEFAULT_INFUSION_DURATION_H,
) -> DosingResult:
    """
    Find optimal dose and interval via grid search.

    Searches all combinations and selects the regimen closest to
    the AUC24/MIC midpoint target while satisfying safety constraints.

    Args:
        params:      Individual PK parameters (post-Bayesian)
        model_type:  PK model type
        target:      PK/PD target thresholds
        doses_mg:    Dose options to search
        intervals_h: Interval options to search
        infusion_h:  Infusion duration (hours)

    Returns:
        DosingResult with optimal regimen
    """
    if target is None:
        target = PKPDTarget()
    if doses_mg is None:
        doses_mg = VANCOMYCIN_DOSES_MG
    if intervals_h is None:
        intervals_h = VANCOMYCIN_INTERVALS_H

    auc_target_mid = (target.auc24_min + target.auc24_max) / 2.0
    best_score = float("inf")
    best_result: dict[str, float] | None = None
    all_results: list[dict[str, float]] = []

    for dose in doses_mg:
        for interval in intervals_h:
            try:
                auc24, trough, peak = _compute_ss_auc24(
                    params, float(dose), float(interval),
                    model_type, infusion_h,
                )
            except (RuntimeError, ValueError):
                continue

            auc24_mic = auc24 / target.mic

            # Score = distance from midpoint target
            score = abs(auc24_mic - auc_target_mid)

            # Penalty for exceeding safety limits
            if trough > target.trough_max:
                score += 1000.0
            if auc24_mic > 800:
                score += 2000.0

            regimen = {
                "dose_mg": float(dose),
                "interval_h": float(interval),
                "auc24": auc24,
                "auc24_mic": auc24_mic,
                "trough": trough,
                "peak": peak,
                "score": score,
            }
            all_results.append(regimen)

            if score < best_score:
                best_score = score
                best_result = regimen

    if best_result is None:
        raise RuntimeError("No valid dosing regimen found")

    # Sort alternatives by score
    all_results.sort(key=lambda x: x["score"])
    top_alternatives = all_results[:5]

    return DosingResult(
        dose_mg=best_result["dose_mg"],
        interval_h=best_result["interval_h"],
        predicted_auc24=best_result["auc24"],
        predicted_auc24_mic=best_result["auc24_mic"],
        predicted_trough=best_result["trough"],
        predicted_peak=best_result["peak"],
        pta=0.0,  # Will be filled by monte_carlo_pta
        infusion_h=infusion_h,
        alternatives=top_alternatives,
    )


# ──────────────────────────────────────────────────────────────────
# Monte Carlo PTA
# ──────────────────────────────────────────────────────────────────

def monte_carlo_pta(
    tv_params: PKParams,
    model: PopPKModel,
    dose_mg: float,
    interval_h: float,
    target: PKPDTarget | None = None,
    n_simulations: int = 5000,
    infusion_h: float = DEFAULT_INFUSION_DURATION_H,
    seed: int | None = None,
) -> float:
    """
    Monte Carlo Probability of Target Attainment (PTA).

    Algorithm:
        1. Sample N=5000 patients from population distribution
        2. For each, simulate PK and compute AUC24/MIC
        3. PTA = fraction achieving target

    Args:
        tv_params:      Typical values (covariate-adjusted)
        model:          PopPK model (for Omega)
        dose_mg:        Dose to evaluate
        interval_h:     Dosing interval
        target:         PK/PD targets
        n_simulations:  Number of Monte Carlo samples
        infusion_h:     Infusion duration
        seed:           Random seed for reproducibility

    Returns:
        PTA as fraction (0-1)
    """
    if target is None:
        target = PKPDTarget()

    rng = np.random.default_rng(seed)
    on_target_count = 0

    for _ in range(n_simulations):
        # Sample individual from population
        ind_params = sample_individual_params(
            tv_params, model.omega_matrix, rng
        )

        # Compute AUC24 at SS
        try:
            auc24, trough, _ = _compute_ss_auc24(
                ind_params, dose_mg, interval_h,
                model.model_type, infusion_h,
            )
        except (RuntimeError, ValueError):
            continue

        auc24_mic = auc24 / target.mic

        # Check if within target
        if (target.auc24_min <= auc24_mic <= target.auc24_max
                and trough <= target.trough_max):
            on_target_count += 1

    return on_target_count / n_simulations
