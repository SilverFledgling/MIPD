"""
PK ODE Solver – Numerical solver for compartmental PK models.

Supports:
    - 1-compartment IV (bolus + infusion)
    - 2-compartment IV (bolus + infusion) [Vancomycin]
    - 1-compartment Oral (first-order absorption)
    - 2-compartment Oral (first-order absorption) [Tacrolimus]

Uses scipy.integrate.solve_ivp with LSODA method.

Reference: Rowland & Tozer, Clinical PK/PD, 5th Ed
Dependencies: numpy, scipy, pk.models
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from pk.models import DoseEvent, ModelType, PKParams, Route


# ──────────────────────────────────────────────────────────────────
# Simulation result
# ──────────────────────────────────────────────────────────────────

class SimulationResult:
    """Container for PK simulation output."""

    def __init__(
        self,
        time: NDArray[np.float64],
        concentration: NDArray[np.float64],
    ) -> None:
        self.time = time
        self.concentration = concentration

    @property
    def auc(self) -> float:
        """Total AUC over entire simulated period (trapezoidal)."""
        return float(np.trapezoid(self.concentration, self.time))

    @property
    def cmax(self) -> float:
        """Maximum concentration observed."""
        return float(np.max(self.concentration))

    @property
    def cmin(self) -> float:
        """Minimum concentration observed (after first dose)."""
        nonzero = self.concentration > 0
        if not np.any(nonzero):
            return 0.0
        return float(np.min(self.concentration[nonzero]))

    def auc_interval(self, t_start: float, t_end: float) -> float:
        """AUC over a specific time interval."""
        mask = (self.time >= t_start) & (self.time <= t_end)
        if np.sum(mask) < 2:
            return 0.0
        return float(np.trapezoid(
            self.concentration[mask], self.time[mask]
        ))

    def concentration_at(self, t: float) -> float:
        """Interpolated concentration at a specific time point."""
        return float(np.interp(t, self.time, self.concentration))


# ──────────────────────────────────────────────────────────────────
# Infusion rate helper
# ──────────────────────────────────────────────────────────────────

def _get_iv_infusion_rate(t: float, doses: list[DoseEvent]) -> float:
    """
    Sum of active IV infusion rates at time t.
    Only considers IV_INFUSION doses with duration > 0.
    """
    rate = 0.0
    for dose in doses:
        if dose.route != Route.IV_INFUSION:
            continue
        if dose.duration <= 0:
            continue
        if dose.time <= t < dose.time + dose.duration:
            rate += dose.amount / dose.duration
    return rate


# ──────────────────────────────────────────────────────────────────
# ODE right-hand-side functions
# ──────────────────────────────────────────────────────────────────

def _rhs_1comp_iv(
    t: float,
    y: list[float],
    params: PKParams,
    doses: list[DoseEvent],
) -> list[float]:
    """
    1-compartment IV: dA/dt = R_inf(t) - ke * A
    y = [A_central]
    """
    a_central = y[0]
    r_inf = _get_iv_infusion_rate(t, doses)
    da_dt = r_inf - params.ke * a_central
    return [da_dt]


def _rhs_2comp_iv(
    t: float,
    y: list[float],
    params: PKParams,
    doses: list[DoseEvent],
) -> list[float]:
    """
    2-compartment IV:
        dA1/dt = R_inf(t) - (CL+Q)*A1/V1 + Q*A2/V2
        dA2/dt = Q*A1/V1 - Q*A2/V2
    y = [A1, A2]
    """
    a1, a2 = y[0], y[1]
    r_inf = _get_iv_infusion_rate(t, doses)
    da1_dt = (
        r_inf
        - (params.CL + params.Q) * a1 / params.V1
        + params.Q * a2 / params.V2
    )
    da2_dt = params.Q * a1 / params.V1 - params.Q * a2 / params.V2
    return [da1_dt, da2_dt]


def _rhs_1comp_oral(
    t: float,
    y: list[float],
    params: PKParams,
    doses: list[DoseEvent],
) -> list[float]:
    """
    1-compartment Oral:
        dA_gut/dt = -Ka * A_gut
        dA/dt     =  Ka * A_gut - ke * A
    y = [A_gut, A_central]
    Oral doses deposited via bolus events into A_gut.
    """
    a_gut, a_central = y[0], y[1]
    da_gut_dt = -params.Ka * a_gut
    da_dt = params.Ka * a_gut - params.ke * a_central
    return [da_gut_dt, da_dt]


def _rhs_2comp_oral(
    t: float,
    y: list[float],
    params: PKParams,
    doses: list[DoseEvent],
) -> list[float]:
    """
    2-compartment Oral:
        dA_gut/dt = -Ka * A_gut
        dA1/dt    =  Ka * A_gut - (CL+Q)*A1/V1 + Q*A2/V2
        dA2/dt    =  Q*A1/V1 - Q*A2/V2
    y = [A_gut, A1, A2]
    """
    a_gut, a1, a2 = y[0], y[1], y[2]
    da_gut_dt = -params.Ka * a_gut
    da1_dt = (
        params.Ka * a_gut
        - (params.CL + params.Q) * a1 / params.V1
        + params.Q * a2 / params.V2
    )
    da2_dt = params.Q * a1 / params.V1 - params.Q * a2 / params.V2
    return [da_gut_dt, da1_dt, da2_dt]


# ──────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────

_ODE_REGISTRY: dict[ModelType, tuple[Callable[..., list[float]], int, int]] = {
    # model_type -> (rhs_function, n_states, central_amount_index)
    ModelType.ONE_COMP_IV: (_rhs_1comp_iv, 1, 0),
    ModelType.TWO_COMP_IV: (_rhs_2comp_iv, 2, 0),
    ModelType.ONE_COMP_ORAL: (_rhs_1comp_oral, 2, 1),
    ModelType.TWO_COMP_ORAL: (_rhs_2comp_oral, 3, 1),
}


# ──────────────────────────────────────────────────────────────────
# Build bolus event schedule
# ──────────────────────────────────────────────────────────────────

def _build_bolus_events(
    doses: list[DoseEvent],
    model_type: ModelType,
    bioavailability: float,
) -> list[tuple[float, int, float]]:
    """
    Build list of (time, compartment_index, amount) for instantaneous
    deposit events (IV bolus or oral dose).

    Returns:
        Sorted list of (time, comp_idx, amount)
    """
    events: list[tuple[float, int, float]] = []

    for dose in doses:
        if dose.route == Route.IV_BOLUS:
            # IV bolus -> deposit into central compartment
            central_idx = 0
            events.append((dose.time, central_idx, dose.amount))

        elif dose.route == Route.ORAL:
            # Oral -> deposit into gut compartment (index 0 for oral models)
            if model_type in (ModelType.ONE_COMP_ORAL, ModelType.TWO_COMP_ORAL):
                gut_idx = 0
                events.append((dose.time, gut_idx, dose.amount * bioavailability))

    events.sort(key=lambda x: x[0])
    return events


# ──────────────────────────────────────────────────────────────────
# Main simulate function
# ──────────────────────────────────────────────────────────────────

def simulate(
    params: PKParams,
    doses: list[DoseEvent],
    model_type: ModelType,
    t_end: float = 72.0,
    dt: float = 0.1,
) -> SimulationResult:
    """
    Simulate PK concentration-time profile.

    Solves the ODE system piecewise, applying bolus events
    (IV bolus or oral deposits) at their scheduled times.

    Args:
        params:     Individual PK parameters
        doses:      List of dose events
        model_type: Which compartmental model to use
        t_end:      Simulation end time (hours)
        dt:         Output time resolution (hours)

    Returns:
        SimulationResult with time and concentration arrays
    """
    # Validate
    errors = params.validate()
    if errors:
        raise ValueError(f"Invalid PK params: {errors}")
    if not doses:
        raise ValueError("At least one dose event is required")

    # Get ODE function and dimensions
    if model_type not in _ODE_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    rhs_func, n_states, central_idx = _ODE_REGISTRY[model_type]

    # Build time grid
    t_eval = np.arange(0.0, t_end + dt, dt)

    # Build bolus events
    bolus_events = _build_bolus_events(doses, model_type, params.F)

    # Collect all event times (unique, sorted)
    event_times_set: set[float] = set()
    for evt_time, _, _ in bolus_events:
        if 0.0 <= evt_time <= t_end:
            event_times_set.add(evt_time)
    segment_boundaries = sorted(event_times_set | {t_end})

    # Solve piecewise
    all_t: list[NDArray[np.float64]] = []
    all_conc: list[NDArray[np.float64]] = []
    y_current = np.zeros(n_states, dtype=np.float64)
    t_current = 0.0
    applied_bolus: set[int] = set()  # Track applied bolus event indices

    for t_boundary in segment_boundaries:
        # Apply bolus events at t_current (only once per event)
        for i, (evt_time, comp_idx, amount) in enumerate(bolus_events):
            if i not in applied_bolus and abs(evt_time - t_current) < 1e-12:
                y_current[comp_idx] += amount
                applied_bolus.add(i)

        # Skip if no time to integrate
        if t_boundary <= t_current:
            continue

        # Time points for this segment
        seg_mask = (t_eval >= t_current) & (t_eval <= t_boundary)
        t_seg = t_eval[seg_mask]
        if len(t_seg) < 2:
            t_seg = np.array([t_current, t_boundary])

        # Capture current rhs_func and params for lambda closure
        current_rhs = rhs_func

        # Solve ODE
        sol = solve_ivp(
            fun=lambda t, y, _rhs=current_rhs: _rhs(t, y, params, doses),
            t_span=(t_current, t_boundary),
            y0=y_current.tolist(),
            t_eval=t_seg,
            method="LSODA",
            rtol=1e-8,
            atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        # Extract central compartment concentration
        central_amount = sol.y[central_idx]
        concentration = central_amount / params.V1

        all_t.append(sol.t)
        all_conc.append(concentration)

        # Update state
        y_current = sol.y[:, -1].copy()
        t_current = t_boundary

    # Concatenate and deduplicate
    t_combined = np.concatenate(all_t)
    c_combined = np.concatenate(all_conc)

    _, unique_idx = np.unique(t_combined, return_index=True)
    t_final = t_combined[unique_idx]
    c_final = c_combined[unique_idx]

    return SimulationResult(time=t_final, concentration=c_final)


# ──────────────────────────────────────────────────────────────────
# Predict at specific observation times (for Bayesian)
# ──────────────────────────────────────────────────────────────────

def predict_concentrations(
    params: PKParams,
    doses: list[DoseEvent],
    obs_times: list[float],
    model_type: ModelType,
) -> NDArray[np.float64]:
    """
    Predict concentrations at specific observation times.
    Used by Bayesian estimators to compute residuals.

    Args:
        params:     Individual PK parameters
        doses:      List of dose events
        obs_times:  List of observation times (hours)
        model_type: Compartmental model type

    Returns:
        Array of predicted concentrations (mg/L)
    """
    if not obs_times:
        return np.array([], dtype=np.float64)

    t_end = max(obs_times) + 1.0
    result = simulate(params, doses, model_type, t_end=t_end, dt=0.05)

    predictions = np.interp(obs_times, result.time, result.concentration)
    return predictions
