"""
Benchmark Metrics — Compute statistical metrics for PK estimation validation.

Single Responsibility: compute_ccc, compute_metrics, compute_rolling_ccc.
Per thuyết minh Công việc 4.2.
"""

from __future__ import annotations

import numpy as np


def compute_ccc(true_arr: np.ndarray, est_arr: np.ndarray) -> float:
    """Concordance Correlation Coefficient (Lin 1989)."""
    if len(true_arr) < 2:
        return np.nan
    mean_t = np.mean(true_arr)
    mean_e = np.mean(est_arr)
    var_t = np.var(true_arr)
    var_e = np.var(est_arr)
    cov_te = np.mean((true_arr - mean_t) * (est_arr - mean_e))
    denom = var_t + var_e + (mean_t - mean_e) ** 2
    return float(2 * cov_te / denom) if denom > 0 else 0.0


def compute_metrics(
    results: list,
    daily_dose: float,
    auc_target_low: float,
    auc_target_high: float,
    omega_prior_diag: np.ndarray,
) -> dict:
    """Compute all benchmark metrics from a list of PatientResult objects.

    Returns dict with keys:
        mpe, mape, rmse, ccc,
        ba_bias, ba_loa_lower, ba_loa_upper,
        coverage_95, target_attain, shrinkage_cl,
        tost_p, tost_equivalent
    """
    n = len(results)
    nan_result = {
        "mpe": np.nan, "mape": np.nan, "rmse": np.nan, "ccc": np.nan,
        "ba_bias": np.nan, "ba_loa_lower": np.nan, "ba_loa_upper": np.nan,
        "coverage_95": np.nan, "target_attain": np.nan, "shrinkage_cl": np.nan,
        "tost_p": np.nan, "tost_equivalent": False,
    }
    if n == 0:
        return nan_result

    true_cls = np.array([r.cl_true for r in results])
    est_cls = np.array([r.cl_est for r in results])

    # ── 1. Core: MPE, MAPE, RMSE, CCC ──
    pe = (est_cls - true_cls) / true_cls * 100
    mpe = float(np.mean(pe))
    mape = float(np.mean(np.abs(pe)))
    rmse = float(np.sqrt(np.mean((est_cls - true_cls) ** 2)))
    ccc = compute_ccc(true_cls, est_cls)

    # ── 2. Bland-Altman ──
    diff = est_cls - true_cls
    ba_bias = float(np.mean(diff))
    ba_sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    ba_loa_lower = ba_bias - 1.96 * ba_sd
    ba_loa_upper = ba_bias + 1.96 * ba_sd

    # ── 3. Coverage 95% ──
    n_with_ci = 0
    n_covered = 0
    for r in results:
        if r.ci_lower is not None and r.ci_upper is not None:
            n_with_ci += 1
            if r.ci_lower <= r.cl_true <= r.ci_upper:
                n_covered += 1
    coverage_95 = (n_covered / n_with_ci * 100) if n_with_ci > 0 else np.nan

    # ── 4. Target Attainment ──
    auc_true = daily_dose / true_cls
    auc_est = daily_dose / est_cls
    in_target_true = (auc_true >= auc_target_low) & (auc_true <= auc_target_high)
    in_target_est = (auc_est >= auc_target_low) & (auc_est <= auc_target_high)
    correct = int((in_target_true == in_target_est).sum())
    target_attain = float(correct / n * 100)

    # ── 5. η-Shrinkage (CL, index 0) ──
    omega_cl = float(omega_prior_diag[0]) if len(omega_prior_diag) > 0 else 0.25
    etas_cl = [
        r.eta_est[0]
        for r in results
        if r.eta_est is not None and len(r.eta_est) > 0
    ]
    if len(etas_cl) >= 2 and omega_cl > 0:
        var_eta_post = float(np.var(etas_cl))
        shrinkage_cl = (1.0 - var_eta_post / omega_cl) * 100
        shrinkage_cl = max(-100.0, min(100.0, shrinkage_cl))
    else:
        shrinkage_cl = np.nan

    # ── 6. TOST Equivalence Test ──
    tost_margin = 20.0
    tost_p = np.nan
    tost_equivalent = False
    if n >= 3:
        try:
            from scipy.stats import ttest_1samp
            t1, p1_two = ttest_1samp(pe, -tost_margin)
            p1 = p1_two / 2 if t1 > 0 else 1.0
            t2, p2_two = ttest_1samp(pe, +tost_margin)
            p2 = p2_two / 2 if t2 < 0 else 1.0
            tost_p = float(max(p1, p2))
            tost_equivalent = tost_p < 0.05
        except ImportError:
            se = float(np.std(pe, ddof=1) / np.sqrt(n))
            if se > 0:
                from math import erfc, sqrt
                t1 = (mpe - (-tost_margin)) / se
                t2 = (mpe - tost_margin) / se
                p1 = 0.5 * erfc(t1 / sqrt(2)) if t1 > 0 else 1.0
                p2 = 0.5 * erfc(-t2 / sqrt(2)) if t2 < 0 else 1.0
                tost_p = float(max(p1, p2))
                tost_equivalent = tost_p < 0.05

    return {
        "mpe": mpe, "mape": mape, "rmse": rmse, "ccc": ccc,
        "ba_bias": ba_bias, "ba_loa_lower": ba_loa_lower, "ba_loa_upper": ba_loa_upper,
        "coverage_95": coverage_95, "target_attain": target_attain,
        "shrinkage_cl": shrinkage_cl,
        "tost_p": tost_p, "tost_equivalent": tost_equivalent,
    }


def compute_rolling_ccc(results: list) -> list[float]:
    """Compute CCC cumulatively after each patient.

    rolling_ccc[i] = CCC using patients 0..i
    """
    rolling = []
    true_so_far = []
    est_so_far = []
    for r in results:
        true_so_far.append(r.cl_true)
        est_so_far.append(r.cl_est)
        if len(true_so_far) >= 3:
            ccc_val = compute_ccc(
                np.array(true_so_far), np.array(est_so_far),
            )
            rolling.append(ccc_val)
        else:
            rolling.append(np.nan)
    return rolling
