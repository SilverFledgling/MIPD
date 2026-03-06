"""
Analytical PK Solutions – Closed-form equations for common PK models.

Used for:
    1. Fast computation when ODE solver is not needed
    2. Validation of ODE solver correctness (cross-check)
    3. Quick a priori predictions

Reference:
    - Rowland & Tozer, Clinical PK/PD, 5th Ed
    - Gibaldi & Perrier, Pharmacokinetics, 2nd Ed

Dependencies: numpy, pk.models
"""

import math

import numpy as np

_trapz = getattr(np, 'trapezoid', None) or np.trapz
from numpy.typing import NDArray

from pk.models import PKParams


# ──────────────────────────────────────────────────────────────────
# 1-Compartment IV Bolus
# ──────────────────────────────────────────────────────────────────

def one_comp_iv_bolus(
    t: NDArray[np.float64],
    dose: float,
    params: PKParams,
) -> NDArray[np.float64]:
    """
    1-compartment IV bolus: C(t) = (Dose / Vd) * exp(-ke * t)

    Args:
        t:      Time points (hours), array
        dose:   Dose amount (mg)
        params: PK parameters (CL, V1 required)

    Returns:
        Concentration array (mg/L)
    """
    return (dose / params.V1) * np.exp(-params.ke * t)


# ──────────────────────────────────────────────────────────────────
# 1-Compartment IV Infusion
# ──────────────────────────────────────────────────────────────────

def one_comp_iv_infusion(
    t: NDArray[np.float64],
    dose: float,
    t_inf: float,
    params: PKParams,
) -> NDArray[np.float64]:
    """
    1-compartment IV infusion.

    During infusion (0 <= t <= T_inf):
        C(t) = (R_inf / CL) * (1 - exp(-ke * t))

    After infusion (t > T_inf):
        C(t) = C(T_inf) * exp(-ke * (t - T_inf))

    Args:
        t:      Time points (hours), array
        dose:   Dose amount (mg)
        t_inf:  Infusion duration (hours)
        params: PK parameters (CL, V1 required)

    Returns:
        Concentration array (mg/L)
    """
    if t_inf <= 0:
        raise ValueError("Infusion duration must be positive")

    r_inf = dose / t_inf
    ke = params.ke
    cl = params.CL

    c = np.zeros_like(t, dtype=np.float64)

    during_mask = t <= t_inf
    after_mask = t > t_inf

    c[during_mask] = (r_inf / cl) * (1.0 - np.exp(-ke * t[during_mask]))

    c_end = (r_inf / cl) * (1.0 - np.exp(-ke * t_inf))
    c[after_mask] = c_end * np.exp(-ke * (t[after_mask] - t_inf))

    return c


# ──────────────────────────────────────────────────────────────────
# 1-Compartment IV Infusion at Steady State
# ──────────────────────────────────────────────────────────────────

def one_comp_iv_infusion_ss(
    t_within_interval: NDArray[np.float64],
    dose: float,
    tau: float,
    t_inf: float,
    params: PKParams,
) -> NDArray[np.float64]:
    """
    1-compartment IV infusion at steady state.

    C_ss,max = (R_inf/CL) * (1-exp(-ke*T_inf)) / (1-exp(-ke*tau))
    C_ss,min = C_ss,max * exp(-ke*(tau-T_inf))

    Args:
        t_within_interval: Time within dosing interval (0 to tau)
        dose:              Dose per interval (mg)
        tau:               Dosing interval (hours)
        t_inf:             Infusion duration (hours)
        params:            PK parameters

    Returns:
        Concentration array (mg/L)
    """
    r_inf = dose / t_inf
    ke = params.ke

    accumulation = 1.0 / (1.0 - math.exp(-ke * tau))

    c = np.zeros_like(t_within_interval, dtype=np.float64)
    during_mask = t_within_interval <= t_inf
    after_mask = t_within_interval > t_inf

    # During infusion phase
    t_dur = t_within_interval[during_mask]
    c[during_mask] = (r_inf / params.CL) * (
        (1.0 - np.exp(-ke * t_dur)) + np.exp(-ke * t_dur)
        * (1.0 - np.exp(-ke * tau)) * (accumulation - 1.0)
    )

    # Simpler: direct SS formula
    # Re-derive: at SS, C at t within interval =
    c_at_end_inf = (r_inf / params.CL) * (
        1.0 - math.exp(-ke * t_inf)
    ) * accumulation
    t_aft = t_within_interval[after_mask]
    c[after_mask] = c_at_end_inf * np.exp(-ke * (t_aft - t_inf))

    # Re-do during infusion with correct SS formula
    c[during_mask] = (r_inf / params.CL) * (
        1.0 - np.exp(-ke * t_dur)
    ) + (r_inf / params.CL) * (1.0 - math.exp(-ke * t_inf)) * (
        accumulation - 1.0
    ) * np.exp(-ke * t_dur)

    return c


# ──────────────────────────────────────────────────────────────────
# 1-Compartment Oral (First-Order Absorption)
# ──────────────────────────────────────────────────────────────────

def one_comp_oral(
    t: NDArray[np.float64],
    dose: float,
    params: PKParams,
) -> NDArray[np.float64]:
    """
    1-compartment oral with first-order absorption.

    C(t) = (F * Dose * Ka) / (Vd * (Ka - ke))
           * [exp(-ke*t) - exp(-Ka*t)]

    Edge case: when Ka ~ ke, use L'Hopital limit.

    Args:
        t:      Time points (hours), array
        dose:   Dose amount (mg)
        params: PK parameters (CL, V1, Ka, F required)

    Returns:
        Concentration array (mg/L)
    """
    f = params.F
    ka = params.Ka
    ke = params.ke
    vd = params.V1

    if ka <= 0:
        raise ValueError("Ka must be positive for oral dosing")

    if abs(ka - ke) < 1e-10:
        # L'Hopital limit: C(t) = (F*Dose/Vd) * ke * t * exp(-ke*t)
        return (f * dose / vd) * ke * t * np.exp(-ke * t)

    return (f * dose * ka) / (vd * (ka - ke)) * (
        np.exp(-ke * t) - np.exp(-ka * t)
    )


# ──────────────────────────────────────────────────────────────────
# Oral: T_max and C_max
# ──────────────────────────────────────────────────────────────────

def oral_tmax(params: PKParams) -> float:
    """
    Time to peak for 1-comp oral model.

    T_max = ln(Ka/ke) / (Ka - ke)

    Args:
        params: PK parameters (Ka, CL, V1 required)

    Returns:
        T_max in hours
    """
    ka = params.Ka
    ke = params.ke

    if ka <= 0 or ke <= 0:
        raise ValueError("Ka and ke must be positive")
    if abs(ka - ke) < 1e-10:
        return 1.0 / ke  # Limit when Ka -> ke

    return math.log(ka / ke) / (ka - ke)


def oral_cmax(dose: float, params: PKParams) -> float:
    """
    Peak concentration for 1-comp oral model.

    C_max = C(T_max)

    Args:
        dose:   Dose amount (mg)
        params: PK parameters

    Returns:
        C_max in mg/L
    """
    t_max = oral_tmax(params)
    t_arr = np.array([t_max])
    c_arr = one_comp_oral(t_arr, dose, params)
    return float(c_arr[0])


# ──────────────────────────────────────────────────────────────────
# 2-Compartment IV Bolus (analytical, for validation)
# ──────────────────────────────────────────────────────────────────

def two_comp_iv_bolus(
    t: NDArray[np.float64],
    dose: float,
    params: PKParams,
) -> NDArray[np.float64]:
    """
    2-compartment IV bolus (bi-exponential).

    C(t) = A * exp(-alpha*t) + B * exp(-beta*t)

    Where alpha, beta are macro rate constants:
        alpha = 0.5 * [(k10+k12+k21) + sqrt((k10+k12+k21)^2 - 4*k10*k21)]
        beta  = 0.5 * [(k10+k12+k21) - sqrt((k10+k12+k21)^2 - 4*k10*k21)]

    Args:
        t:      Time points (hours)
        dose:   Dose amount (mg)
        params: PK parameters (CL, V1, Q, V2 required)

    Returns:
        Concentration array (mg/L)
    """
    k10 = params.CL / params.V1
    k12 = params.Q / params.V1
    k21 = params.Q / params.V2

    sum_k = k10 + k12 + k21
    discriminant = sum_k ** 2 - 4.0 * k10 * k21

    if discriminant < 0:
        raise ValueError("Negative discriminant in 2-comp model")

    sqrt_disc = math.sqrt(discriminant)
    alpha = 0.5 * (sum_k + sqrt_disc)
    beta = 0.5 * (sum_k - sqrt_disc)

    coeff_a = (dose / params.V1) * (alpha - k21) / (alpha - beta)
    coeff_b = (dose / params.V1) * (k21 - beta) / (alpha - beta)

    return coeff_a * np.exp(-alpha * t) + coeff_b * np.exp(-beta * t)


# ──────────────────────────────────────────────────────────────────
# AUC (trapezoidal rule)
# ──────────────────────────────────────────────────────────────────

def auc_trapezoidal(
    t: NDArray[np.float64],
    c: NDArray[np.float64],
) -> float:
    """
    AUC by linear trapezoidal rule.

    AUC = sum_i [(C_i + C_{i+1}) / 2] * (t_{i+1} - t_i)

    Args:
        t: Time points (hours)
        c: Concentration values (mg/L)

    Returns:
        AUC in mg*h/L
    """
    return float(_trapz(c, t))


def auc24_from_cl(daily_dose: float, cl: float) -> float:
    """
    AUC24 at steady state from clearance (1-comp approximation).

    AUC24_ss = Daily_Dose / CL

    Args:
        daily_dose: Total daily dose (mg/day)
        cl:         Clearance (L/h)

    Returns:
        AUC24 in mg*h/L
    """
    if cl <= 0:
        raise ValueError("CL must be positive")
    return daily_dose / cl
