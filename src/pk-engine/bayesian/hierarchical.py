"""
Hierarchical Bayesian Model – 3-tier global→local→individual.

Implements the Hierarchical Bayesian model from mathematical_reference §4.11
for combining international (global) PK data with Vietnamese (local) data.

Architecture (3-tier):
    Tier 3 (Hyperpriors / Global):
        μ_local ~ LogNormal(log(μ_global), τ²)
        ω_local ~ HalfNormal(ω_global_scale)

    Tier 2 (Local Population / Vietnam):
        η_i | μ_local, ω_local ~ Normal(0, ω_local)
        θ_i = μ_local * exp(η_i)

    Tier 1 (Observation / Likelihood):
        y_ij | θ_i ~ Normal(f(θ_i, t_ij), σ²)

Key concept (Partial Pooling):
    When local data is sparse → posterior μ_local ≈ μ_global  (shrinkage)
    When local data is abundant → posterior μ_local diverges  (adaptation)

Reference:
    - Gelman et al. (2013), Bayesian Data Analysis, Ch. 5
    - Plan §4.11: Hierarchical Bayesian (3-tier)
    - Công việc 2.2 in NAFOSTED proposal

Dependencies: jax, numpyro, numpy, pk.models
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pk.models import (
    DoseEvent,
    ErrorModel,
    ModelType,
    Observation,
    PKParams,
    PopPKModel,
)
from pk.population import apply_iiv

# Lazy imports for JAX/NumPyro
_NUMPYRO_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC as _MCMC, NUTS as _NUTS
    _NUMPYRO_AVAILABLE = True
except ImportError:
    pass


# Re-use the JAX-native PK solver from mcmc module
from bayesian.mcmc import _jax_predict_2comp_iv, _compute_rhat, _compute_ess


# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────

@dataclass
class PatientRecord:
    """
    Data for one patient in the hierarchical model.

    Attributes:
        doses:          List of dose events for this patient
        observations:   List of TDM observations for this patient
        patient_id:     Optional patient identifier
    """
    doses: list[DoseEvent]
    observations: list[Observation]
    patient_id: str = ""


@dataclass
class HierarchicalResult:
    """
    Result of Hierarchical Bayesian inference.

    Attributes:
        mu_local:           Posterior local population means {CL, V1, Q, V2}
        omega_local:        Posterior local population variances {CL, V1, Q, V2}
        individual_params:  Per-patient posterior PK parameters
        pooling_ratio:      How much local deviates from global (0=global, 1=local)
        summary:            Full posterior summary
        rhat:               Gelman-Rubin R-hat per parameter
        ess:                Effective sample size per parameter
        n_samples:          Total posterior samples
        n_chains:           Number of chains
        n_divergences:      Number of divergent transitions
        converged:          True if all R-hat < 1.05 and n_divergences == 0
        n_patients:         Number of patients in the local dataset
    """
    mu_local: dict[str, dict[str, float]]
    omega_local: dict[str, dict[str, float]]
    individual_params: list[dict[str, dict[str, float]]]
    pooling_ratio: dict[str, float]
    summary: dict[str, dict[str, float]]
    rhat: dict[str, float]
    ess: dict[str, float]
    n_samples: int
    n_chains: int
    n_divergences: int
    converged: bool
    n_patients: int


# ──────────────────────────────────────────────────────────────────
# NumPyro hierarchical model (3-tier, JAX-native)
# ──────────────────────────────────────────────────────────────────

def _hierarchical_model(
    # Tier 3: Global priors (known constants)
    mu_global_cl, mu_global_v1, mu_global_q, mu_global_v2,
    tau_mu,  # how much local can deviate from global
    # Per-patient data (lists of JAX arrays)
    all_dose_times, all_dose_amounts, all_dose_durations,
    all_obs_times,
    # Error model
    sigma_prop, sigma_add,
    # Number of patients
    n_patients,
    # Observations (conditioning)
    all_y_obs=None,
):
    """
    NumPyro hierarchical model for population-level Bayesian inference.

    Tier 3: μ_local ~ LogNormal(log(μ_global), τ²)
    Tier 2: η_i ~ Normal(0, ω_local)  →  θ_i = μ_local * exp(η_i)
    Tier 1: y_ij ~ Normal(f(θ_i, t_ij), σ²)
    """

    # ── Tier 3: Hyperpriors (Global → Local population) ──
    # μ_local is sampled around μ_global with uncertainty τ
    log_cl_local = numpyro.sample(
        "log_CL_local",
        dist.Normal(jnp.log(mu_global_cl), tau_mu),
    )
    log_v1_local = numpyro.sample(
        "log_V1_local",
        dist.Normal(jnp.log(mu_global_v1), tau_mu),
    )
    log_q_local = numpyro.sample(
        "log_Q_local",
        dist.Normal(jnp.log(mu_global_q), tau_mu),
    )
    log_v2_local = numpyro.sample(
        "log_V2_local",
        dist.Normal(jnp.log(mu_global_v2), tau_mu),
    )

    mu_cl_local = jnp.exp(log_cl_local)
    mu_v1_local = jnp.exp(log_v1_local)
    mu_q_local = jnp.exp(log_q_local)
    mu_v2_local = jnp.exp(log_v2_local)

    # Local population variability (ω²_local) — sampled with HalfNormal prior
    omega_cl = numpyro.sample("omega_CL", dist.HalfNormal(0.5))
    omega_v1 = numpyro.sample("omega_V1", dist.HalfNormal(0.5))

    # ── Tier 2 + Tier 1: For each patient ──
    for i in range(n_patients):
        # Tier 2: Individual random effects
        eta_cl_i = numpyro.sample(
            f"eta_CL_{i}", dist.Normal(0.0, omega_cl),
        )
        eta_v1_i = numpyro.sample(
            f"eta_V1_{i}", dist.Normal(0.0, omega_v1),
        )

        # Individual parameters: θ_i = μ_local * exp(η_i)
        cl_i = mu_cl_local * jnp.exp(eta_cl_i)
        v1_i = mu_v1_local * jnp.exp(eta_v1_i)
        q_i = mu_q_local
        v2_i = mu_v2_local

        # Tier 1: Predict concentrations and compare to observations
        c_pred = _jax_predict_2comp_iv(
            cl_i, v1_i, q_i, v2_i,
            all_dose_times[i],
            all_dose_amounts[i],
            all_dose_durations[i],
            all_obs_times[i],
        )

        n_obs_i = all_obs_times[i].shape[0]
        for j in range(n_obs_i):
            c_j = jnp.maximum(c_pred[j], 1e-10)
            sd_j = jnp.sqrt((sigma_prop * c_j) ** 2 + sigma_add ** 2)
            sd_j = jnp.maximum(sd_j, 1e-6)

            obs_val = None
            if all_y_obs is not None:
                obs_val = all_y_obs[i][j]

            numpyro.sample(
                f"obs_{i}_{j}",
                dist.Normal(c_j, sd_j),
                obs=obs_val,
            )


# ──────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────

def run_hierarchical(
    model: PopPKModel,
    global_params: PKParams,
    patients: list[PatientRecord],
    tau_mu: float = 0.3,
    n_warmup: int = 300,
    n_samples: int = 500,
    n_chains: int = 2,
    seed: int = 42,
    target_accept_prob: float = 0.80,
    max_tree_depth: int = 8,
) -> HierarchicalResult:
    """
    Run Hierarchical Bayesian inference (3-tier model).

    This combines global PK knowledge with local patient data to estimate
    population-level parameters specific to Vietnamese patients.

    Args:
        model:              PopPK model (for error model specification)
        global_params:      Global typical values (from international literature)
        patients:           List of PatientRecord (local VN patients)
        tau_mu:             Prior std for how much local can deviate from global.
                            Small τ → strong shrinkage to global.
                            Large τ → local data dominates.
        n_warmup:           Warmup iterations per chain
        n_samples:          Post-warmup samples per chain
        n_chains:           Number of MCMC chains
        seed:               Random seed
        target_accept_prob: NUTS target acceptance probability
        max_tree_depth:     NUTS max tree depth

    Returns:
        HierarchicalResult with posterior estimates for local population
        and individual parameters.
    """
    if not _NUMPYRO_AVAILABLE:
        raise RuntimeError(
            "NumPyro is required for Hierarchical Bayesian. "
            "Install with: pip install jax[cpu] numpyro"
        )

    if not patients:
        raise ValueError("At least one patient record is required")

    for i, p in enumerate(patients):
        if not p.doses:
            raise ValueError(f"Patient {i} has no dose events")
        if not p.observations:
            raise ValueError(f"Patient {i} has no observations")

    n_patients = len(patients)

    # Prepare per-patient data as JAX arrays
    all_dose_times = []
    all_dose_amounts = []
    all_dose_durations = []
    all_obs_times = []
    all_y_obs = []

    for p in patients:
        all_dose_times.append(
            jnp.array([d.time for d in p.doses], dtype=jnp.float32)
        )
        all_dose_amounts.append(
            jnp.array([d.amount for d in p.doses], dtype=jnp.float32)
        )
        all_dose_durations.append(
            jnp.array(
                [d.duration if d.duration > 0 else 0.01 for d in p.doses],
                dtype=jnp.float32,
            )
        )
        all_obs_times.append(
            jnp.array([o.time for o in p.observations], dtype=jnp.float32)
        )
        all_y_obs.append(
            jnp.array([o.concentration for o in p.observations],
                       dtype=jnp.float32)
        )

    # Error model parameters
    sigma_prop = float(model.error_model.sigma_prop)
    sigma_add = float(model.error_model.sigma_add)

    # Run MCMC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        kernel = _NUTS(
            _hierarchical_model,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
        )

        mcmc = _MCMC(
            kernel,
            num_warmup=n_warmup,
            num_samples=n_samples,
            num_chains=n_chains,
            progress_bar=False,
        )

        rng_key = jax.random.PRNGKey(seed)
        mcmc.run(
            rng_key,
            mu_global_cl=float(global_params.CL),
            mu_global_v1=float(global_params.V1),
            mu_global_q=float(global_params.Q),
            mu_global_v2=float(global_params.V2),
            tau_mu=float(tau_mu),
            all_dose_times=all_dose_times,
            all_dose_amounts=all_dose_amounts,
            all_dose_durations=all_dose_durations,
            all_obs_times=all_obs_times,
            sigma_prop=sigma_prop,
            sigma_add=sigma_add,
            n_patients=n_patients,
            all_y_obs=all_y_obs,
        )

    # Extract posterior samples
    samples = mcmc.get_samples(group_by_chain=True)

    pk_names = ["CL", "V1", "Q", "V2"]
    log_names = [f"log_{p}_local" for p in pk_names]

    # ── Posterior summaries for local population means (Tier 3 output) ──
    mu_local_summary: dict[str, dict[str, float]] = {}
    rhat_dict: dict[str, float] = {}
    ess_dict: dict[str, float] = {}
    summary_dict: dict[str, dict[str, float]] = {}

    for pk_name, log_name in zip(pk_names, log_names):
        if log_name in samples:
            chains = np.array(samples[log_name])  # (n_chains, n_samples)
            rhat_dict[pk_name] = _compute_rhat(chains)

            all_log = chains.flatten()
            ess_dict[pk_name] = _compute_ess(all_log)

            # Transform from log-scale to natural scale
            all_mu = np.exp(all_log)

            mu_local_summary[pk_name] = {
                "mean": float(np.mean(all_mu)),
                "sd": float(np.std(all_mu, ddof=1)),
                "median": float(np.median(all_mu)),
                "ci95_lower": float(np.percentile(all_mu, 2.5)),
                "ci95_upper": float(np.percentile(all_mu, 97.5)),
            }

            summary_dict[f"mu_{pk_name}_local"] = mu_local_summary[pk_name]
        else:
            global_val = getattr(global_params, pk_name)
            mu_local_summary[pk_name] = {
                "mean": global_val, "sd": 0.0, "median": global_val,
                "ci95_lower": global_val, "ci95_upper": global_val,
            }
            rhat_dict[pk_name] = 1.0
            ess_dict[pk_name] = float(n_samples * n_chains)

    # ── Posterior summaries for local omega (Tier 3 output) ──
    omega_local_summary: dict[str, dict[str, float]] = {}
    for pk_name in ["CL", "V1"]:
        omega_name = f"omega_{pk_name}"
        if omega_name in samples:
            chains = np.array(samples[omega_name])
            all_omega = chains.flatten()
            omega_local_summary[pk_name] = {
                "mean": float(np.mean(all_omega)),
                "sd": float(np.std(all_omega, ddof=1)),
                "median": float(np.median(all_omega)),
                "ci95_lower": float(np.percentile(all_omega, 2.5)),
                "ci95_upper": float(np.percentile(all_omega, 97.5)),
            }
        else:
            omega_local_summary[pk_name] = {
                "mean": 0.0, "sd": 0.0, "median": 0.0,
                "ci95_lower": 0.0, "ci95_upper": 0.0,
            }

    # ── Per-patient individual parameters (Tier 2 output) ──
    individual_params: list[dict[str, dict[str, float]]] = []
    for i in range(n_patients):
        patient_params: dict[str, dict[str, float]] = {}
        for pk_name in ["CL", "V1"]:
            eta_name = f"eta_{pk_name}_{i}"
            log_mu_name = f"log_{pk_name}_local"
            if eta_name in samples and log_mu_name in samples:
                eta_chains = np.array(samples[eta_name])
                mu_chains = np.array(samples[log_mu_name])
                # θ_i = μ_local * exp(η_i)
                theta_samples = np.exp(mu_chains) * np.exp(eta_chains)
                all_theta = theta_samples.flatten()
                patient_params[pk_name] = {
                    "mean": float(np.mean(all_theta)),
                    "sd": float(np.std(all_theta, ddof=1)),
                    "median": float(np.median(all_theta)),
                    "ci95_lower": float(np.percentile(all_theta, 2.5)),
                    "ci95_upper": float(np.percentile(all_theta, 97.5)),
                }
            else:
                global_val = getattr(global_params, pk_name)
                patient_params[pk_name] = {
                    "mean": global_val, "sd": 0.0, "median": global_val,
                    "ci95_lower": global_val, "ci95_upper": global_val,
                }
        # Q and V2 are at population level (no IIV sampled for them)
        for pk_name in ["Q", "V2"]:
            patient_params[pk_name] = mu_local_summary.get(pk_name, {
                "mean": getattr(global_params, pk_name),
            })
        individual_params.append(patient_params)

    # ── Pooling ratio: how much local diverged from global ──
    pooling_ratio: dict[str, float] = {}
    for pk_name in pk_names:
        global_val = getattr(global_params, pk_name)
        local_mean = mu_local_summary[pk_name]["mean"]
        local_sd = mu_local_summary[pk_name]["sd"]
        if local_sd > 0 and global_val > 0:
            # Relative deviation: |μ_local - μ_global| / μ_global
            deviation = abs(local_mean - global_val) / global_val
            # Normalize to [0, 1]: 0 = identical to global, 1 = fully local
            pooling_ratio[pk_name] = min(deviation / tau_mu, 1.0)
        else:
            pooling_ratio[pk_name] = 0.0

    # ── Divergences and convergence ──
    n_divergences = 0
    try:
        extra_fields = mcmc.get_extra_fields(group_by_chain=False)
        if "diverging" in extra_fields:
            n_divergences = int(np.sum(np.array(extra_fields["diverging"])))
    except Exception:
        pass

    converged = (
        all(r < 1.05 for r in rhat_dict.values() if not np.isnan(r))
        and n_divergences == 0
    )

    return HierarchicalResult(
        mu_local=mu_local_summary,
        omega_local=omega_local_summary,
        individual_params=individual_params,
        pooling_ratio=pooling_ratio,
        summary=summary_dict,
        rhat=rhat_dict,
        ess=ess_dict,
        n_samples=n_samples * n_chains,
        n_chains=n_chains,
        n_divergences=n_divergences,
        converged=converged,
        n_patients=n_patients,
    )
