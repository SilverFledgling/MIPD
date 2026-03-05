"""
VPC Simulation – Generate data for Visual Predictive Check.

Protocol:
    1. For each original patient, sample N=200 virtual patients from Omega
    2. Simulate PK for each virtual patient with same doses
    3. Compute percentiles (5th, 50th, 95th) at each time point
    4. Compare observed data against simulated percentile bands

Reference: Holford (2005), PAGE Meeting
Dependencies: numpy, pk.models, pk.solver, pk.population
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from pk.models import DoseEvent, ModelType, PKParams, PopPKModel
from pk.solver import simulate
from pk.population import sample_individual_params


# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────

@dataclass
class VPCData:
    """
    VPC simulation data for plotting.

    Attributes:
        time_grid:     Time points for percentile curves (hours)
        obs_pctile_5:  5th percentile of observed (if available)
        obs_pctile_50: 50th percentile of observed (median)
        obs_pctile_95: 95th percentile of observed
        sim_pctile_5:  5th percentile of simulated
        sim_pctile_50: 50th percentile of simulated (median)
        sim_pctile_95: 95th percentile of simulated
        sim_ci_5:      95% CI of simulated 5th percentile [lower, upper]
        sim_ci_50:     95% CI of simulated median [lower, upper]
        sim_ci_95:     95% CI of simulated 95th percentile [lower, upper]
        n_simulations: Number of virtual patients per original
    """
    time_grid: NDArray[np.float64]
    sim_pctile_5: NDArray[np.float64]
    sim_pctile_50: NDArray[np.float64]
    sim_pctile_95: NDArray[np.float64]
    sim_ci_5: tuple[NDArray[np.float64], NDArray[np.float64]]
    sim_ci_50: tuple[NDArray[np.float64], NDArray[np.float64]]
    sim_ci_95: tuple[NDArray[np.float64], NDArray[np.float64]]
    n_simulations: int


# ──────────────────────────────────────────────────────────────────
# VPC simulation
# ──────────────────────────────────────────────────────────────────

def simulate_vpc(
    tv_params: PKParams,
    model: PopPKModel,
    doses: list[DoseEvent],
    model_type: ModelType | None = None,
    t_end: float = 48.0,
    dt: float = 0.5,
    n_simulations: int = 200,
    n_replicates: int = 100,
    seed: int | None = None,
) -> VPCData:
    """
    Generate VPC simulation data.

    Algorithm:
        1. Simulate N virtual patients from population
        2. For each replicate, compute percentiles
        3. Compute CI of percentiles across replicates

    Args:
        tv_params:       Typical values (covariate-adjusted)
        model:           PopPK model
        doses:           Dose schedule
        model_type:      PK model type
        t_end:           Simulation end (hours)
        dt:              Time step for grid (hours)
        n_simulations:   Patients per replicate
        n_replicates:    Number of replicates for CI
        seed:            Random seed

    Returns:
        VPCData with percentile curves and CIs
    """
    if model_type is None:
        model_type = model.model_type

    rng = np.random.default_rng(seed)
    time_grid = np.arange(0, t_end + dt, dt)
    n_times = len(time_grid)

    # Store percentiles from each replicate
    pctile_5_reps = np.zeros((n_replicates, n_times))
    pctile_50_reps = np.zeros((n_replicates, n_times))
    pctile_95_reps = np.zeros((n_replicates, n_times))

    for rep in range(n_replicates):
        conc_matrix = np.zeros((n_simulations, n_times))

        for i in range(n_simulations):
            # Sample individual from population
            ind_params = sample_individual_params(
                tv_params, model.omega_matrix, rng
            )

            try:
                result = simulate(
                    ind_params, doses, model_type, t_end=t_end, dt=dt
                )
                # Interpolate to common time grid
                conc_matrix[i, :] = np.interp(
                    time_grid, result.time, result.concentration
                )
            except (RuntimeError, ValueError):
                # Failed simulation: use zeros (excluded in percentile calc)
                conc_matrix[i, :] = np.nan

        # Compute percentiles ignoring NaN
        for t_idx in range(n_times):
            col = conc_matrix[:, t_idx]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                pctile_5_reps[rep, t_idx] = float(np.percentile(valid, 5))
                pctile_50_reps[rep, t_idx] = float(np.percentile(valid, 50))
                pctile_95_reps[rep, t_idx] = float(np.percentile(valid, 95))

    # Overall percentiles (median of replicates)
    sim_pctile_5 = np.median(pctile_5_reps, axis=0)
    sim_pctile_50 = np.median(pctile_50_reps, axis=0)
    sim_pctile_95 = np.median(pctile_95_reps, axis=0)

    # CI of percentiles (2.5th and 97.5th of replicates)
    sim_ci_5 = (
        np.percentile(pctile_5_reps, 2.5, axis=0),
        np.percentile(pctile_5_reps, 97.5, axis=0),
    )
    sim_ci_50 = (
        np.percentile(pctile_50_reps, 2.5, axis=0),
        np.percentile(pctile_50_reps, 97.5, axis=0),
    )
    sim_ci_95 = (
        np.percentile(pctile_95_reps, 2.5, axis=0),
        np.percentile(pctile_95_reps, 97.5, axis=0),
    )

    return VPCData(
        time_grid=time_grid,
        sim_pctile_5=sim_pctile_5,
        sim_pctile_50=sim_pctile_50,
        sim_pctile_95=sim_pctile_95,
        sim_ci_5=sim_ci_5,
        sim_ci_50=sim_ci_50,
        sim_ci_95=sim_ci_95,
        n_simulations=n_simulations,
    )
