"""
MCMC Estimator – Markov Chain Monte Carlo posterior sampling via NumPyro NUTS.

Implements full Bayesian inference for individual PK parameters using
a JAX-native PK solver (no scipy dependency inside the model).

Architecture:
    - NumPyro model samples eta ~ N(0, Omega)
    - JAX-native 2-comp IV solver predicts concentrations (superposition)
    - Likelihood: C_obs ~ N(C_pred, sigma)
    - NUTS samples the posterior

The key design decision: the PK solver MUST be written in JAX (jnp)
because NumPyro traces through the model with abstract JAX values.
scipy/numpy ODE solvers cannot be used inside the model function.

Reference:
    - Hoffman & Gelman (2014), JMLR 15(1), 1593-1623 (NUTS)
    - NumPyro documentation: https://num.pyro.ai/

Dependencies: jax, numpyro, numpy, pk.models, pk.population
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pk.models import (
    DoseEvent,
    ErrorModel,
    ModelType,
    Observation,
    PKParams,
    PopPKModel,
    Route,
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


# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────

@dataclass
class MCMCResult:
    """
    Result of MCMC posterior sampling.

    Attributes:
        posterior_eta:     Posterior samples for eta (n_samples, n_params)
        posterior_params:  Summary of PK params (mean + CI)
        summary:           Dict of {param_name: {mean, sd, ci95_lower, ...}}
        rhat:              Gelman-Rubin R-hat per parameter
        ess:               Effective sample size per parameter
        n_samples:         Total posterior samples (after warmup)
        n_chains:          Number of chains
        n_divergences:     Number of divergent transitions
        converged:         True if all R-hat < 1.05 and n_divergences == 0
        map_params:        MAP estimate (posterior mean)
    """
    posterior_eta: NDArray[np.float64]
    posterior_params: dict[str, dict[str, float]]
    summary: dict[str, dict[str, float]]
    rhat: dict[str, float]
    ess: dict[str, float]
    n_samples: int
    n_chains: int
    n_divergences: int
    converged: bool
    map_params: PKParams


# ──────────────────────────────────────────────────────────────────
# JAX-native 2-compartment IV model (superposition principle)
# ──────────────────────────────────────────────────────────────────

def _jax_predict_2comp_iv(
    cl: float, v1: float, q: float, v2: float,
    dose_times: jnp.ndarray,
    dose_amounts: jnp.ndarray,
    dose_durations: jnp.ndarray,
    obs_times: jnp.ndarray,
) -> jnp.ndarray:
    """
    JAX-native 2-compartment IV model using superposition.

    For IV infusion of duration tau:
        C(t) = (R0/V1) * [A/alpha * (1-exp(-alpha*tau)) * exp(-alpha*(t-t0-tau))
                         + B/beta  * (1-exp(-beta*tau))  * exp(-beta*(t-t0-tau))]
        where R0 = dose/tau, and A, B, alpha, beta from micro-constants.

    Uses jnp throughout so JAX can trace and differentiate.
    """
    ke = cl / v1
    k12 = q / v1
    k21 = q / jnp.maximum(v2, 1e-10)

    # Eigenvalues (alpha, beta)
    sum_k = ke + k12 + k21
    prod_k = ke * k21
    discriminant = jnp.maximum(sum_k ** 2 - 4.0 * prod_k, 1e-20)
    sqrt_disc = jnp.sqrt(discriminant)

    alpha = (sum_k + sqrt_disc) / 2.0
    beta = (sum_k - sqrt_disc) / 2.0

    # Macro-constants A and B
    A = (alpha - k21) / (alpha - beta)
    B = (k21 - beta) / (alpha - beta)

    n_obs = obs_times.shape[0]
    n_doses = dose_times.shape[0]

    c_total = jnp.zeros(n_obs)

    # Superposition: sum contributions from each dose
    def add_dose(carry, dose_idx):
        c_accum = carry
        t_dose = dose_times[dose_idx]
        amount = dose_amounts[dose_idx]
        duration = dose_durations[dose_idx]

        # IV infusion (duration > 0)
        # Use infusion formula + bolus formula blended by duration
        # For very short duration, approaches bolus
        safe_dur = jnp.maximum(duration, 1e-6)
        rate = amount / safe_dur  # mg/h

        def compute_conc_at_obs(c_prev, obs_idx):
            t = obs_times[obs_idx]
            dt = t - t_dose

            # Only contribute if t >= t_dose
            # During infusion (0 <= dt <= duration):
            c_during = (rate / v1) * (
                A / alpha * (1.0 - jnp.exp(-alpha * jnp.minimum(dt, safe_dur)))
                + B / beta * (1.0 - jnp.exp(-beta * jnp.minimum(dt, safe_dur)))
            )

            # After infusion (dt > duration):
            c_after = (rate / v1) * (
                A / alpha * (1.0 - jnp.exp(-alpha * safe_dur))
                * jnp.exp(-alpha * (dt - safe_dur))
                + B / beta * (1.0 - jnp.exp(-beta * safe_dur))
                * jnp.exp(-beta * (dt - safe_dur))
            )

            # Select based on phase
            is_during = (dt >= 0) & (dt <= safe_dur)
            is_after = dt > safe_dur
            c_dose = jnp.where(is_during, c_during,
                               jnp.where(is_after, c_after, 0.0))

            return c_prev + c_dose, None

        # Scan over observations for this dose
        new_c, _ = jax.lax.scan(
            compute_conc_at_obs,
            jnp.float32(0.0),  # dummy init
            jnp.arange(n_obs),
        )
        # Actually we need per-observation, use vmap instead
        return c_accum, None

    # Simpler approach: vectorized over obs and doses
    def single_dose_contribution(dose_idx):
        t_dose = dose_times[dose_idx]
        amount = dose_amounts[dose_idx]
        duration = dose_durations[dose_idx]
        safe_dur = jnp.maximum(duration, 1e-6)
        rate = amount / safe_dur

        def conc_at_time(t):
            dt = t - t_dose

            c_during = (rate / v1) * (
                A / alpha * (1.0 - jnp.exp(-alpha * jnp.clip(dt, 0, safe_dur)))
                + B / beta * (1.0 - jnp.exp(-beta * jnp.clip(dt, 0, safe_dur)))
            )

            c_after = (rate / v1) * (
                A / alpha * (1.0 - jnp.exp(-alpha * safe_dur))
                * jnp.exp(-alpha * jnp.maximum(dt - safe_dur, 0.0))
                + B / beta * (1.0 - jnp.exp(-beta * safe_dur))
                * jnp.exp(-beta * jnp.maximum(dt - safe_dur, 0.0))
            )

            is_during = (dt >= 0) & (dt <= safe_dur)
            is_after = dt > safe_dur
            return jnp.where(is_during, c_during,
                             jnp.where(is_after, c_after, 0.0))

        return jax.vmap(conc_at_time)(obs_times)  # (n_obs,)

    # Sum over all doses
    all_contributions = jax.vmap(single_dose_contribution)(
        jnp.arange(n_doses)
    )  # (n_doses, n_obs)

    return jnp.sum(all_contributions, axis=0)  # (n_obs,)


# ──────────────────────────────────────────────────────────────────
# NumPyro probabilistic model (fully JAX-native)
# ──────────────────────────────────────────────────────────────────

def _pk_model_numpyro(
    omega_diag,
    tv_cl, tv_v1, tv_q, tv_v2,
    dose_times, dose_amounts, dose_durations,
    obs_times,
    sigma_prop, sigma_add,
    y_obs=None,
):
    """
    NumPyro model for PK parameter estimation (JAX-native).

    All operations use jnp so JAX tracing works correctly.
    """
    n_params = omega_diag.shape[0]

    # Prior: eta ~ N(0, omega)
    eta_CL = numpyro.sample("eta_CL", dist.Normal(0.0, jnp.sqrt(omega_diag[0])))
    eta_V1 = numpyro.sample("eta_V1", dist.Normal(0.0, jnp.sqrt(omega_diag[1])))

    if n_params >= 3:
        eta_Q = numpyro.sample("eta_Q", dist.Normal(0.0, jnp.sqrt(omega_diag[2])))
    else:
        eta_Q = 0.0

    if n_params >= 4:
        eta_V2 = numpyro.sample("eta_V2", dist.Normal(0.0, jnp.sqrt(omega_diag[3])))
    else:
        eta_V2 = 0.0

    # Individual parameters: theta = TV * exp(eta)
    cl = tv_cl * jnp.exp(eta_CL)
    v1 = tv_v1 * jnp.exp(eta_V1)
    q = tv_q * jnp.exp(eta_Q)
    v2 = tv_v2 * jnp.exp(eta_V2)

    # Predict concentrations (JAX-native)
    c_pred = _jax_predict_2comp_iv(
        cl, v1, q, v2,
        dose_times, dose_amounts, dose_durations, obs_times,
    )

    # Observation model: combined error
    n_obs = obs_times.shape[0]
    for j in range(n_obs):
        c_j = jnp.maximum(c_pred[j], 1e-10)
        sd_j = jnp.sqrt((sigma_prop * c_j) ** 2 + sigma_add ** 2)
        sd_j = jnp.maximum(sd_j, 1e-6)

        numpyro.sample(
            f"obs_{j}",
            dist.Normal(c_j, sd_j),
            obs=y_obs[j] if y_obs is not None else None,
        )


# ──────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────

def _compute_rhat(chains: NDArray[np.float64]) -> float:
    """Gelman-Rubin R-hat. Target: < 1.05."""
    n_chains, n_samples = chains.shape
    if n_chains < 2 or n_samples < 2:
        return float("nan")

    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means)
    B = n_samples / (n_chains - 1) * np.sum((chain_means - overall_mean) ** 2)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    var_hat = (1 - 1 / n_samples) * W + B / n_samples

    if W <= 0:
        return float("nan")
    return float(np.sqrt(var_hat / W))


def _compute_ess(samples: NDArray[np.float64]) -> float:
    """Effective sample size via autocorrelation."""
    n = len(samples)
    if n < 4:
        return float(n)

    x = samples - np.mean(samples)
    var = np.var(x, ddof=0)
    if var == 0:
        return float(n)

    fft_x = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n]
    acf /= acf[0]

    tau = 1.0
    for k in range(1, n // 2):
        rho_pair = acf[2 * k - 1] + acf[2 * k]
        if rho_pair < 0:
            break
        tau += 2 * rho_pair

    return max(float(n / tau), 1.0)


# ──────────────────────────────────────────────────────────────────
# Main MCMC runner
# ──────────────────────────────────────────────────────────────────

def run_mcmc(
    model: PopPKModel,
    tv_params: PKParams,
    doses: list[DoseEvent],
    observations: list[Observation],
    model_type: ModelType | None = None,
    n_warmup: int = 500,
    n_samples: int = 1000,
    n_chains: int = 2,
    seed: int = 42,
    target_accept_prob: float = 0.80,
    max_tree_depth: int = 10,
) -> MCMCResult:
    """
    Run MCMC posterior sampling using NUTS.

    Args:
        model:                PopPK model (for Omega and error model)
        tv_params:            Covariate-adjusted typical values
        doses:                Dose administration events
        observations:         TDM concentration measurements
        model_type:           PK model type (defaults to model.model_type)
        n_warmup:             Warmup iterations per chain
        n_samples:            Post-warmup samples per chain
        n_chains:             Number of chains
        seed:                 Random seed
        target_accept_prob:   NUTS target acceptance (0.8 default)
        max_tree_depth:       NUTS max tree depth

    Returns:
        MCMCResult with posterior samples and diagnostics
    """
    if not _NUMPYRO_AVAILABLE:
        raise RuntimeError(
            "NumPyro is required for MCMC. "
            "Install with: pip install jax[cpu] numpyro"
        )

    if not observations:
        raise ValueError("At least one observation is required for MCMC")
    if not doses:
        raise ValueError("At least one dose event is required for MCMC")

    if model_type is None:
        model_type = model.model_type

    # Prepare data as JAX arrays
    omega = np.array(model.omega_matrix, dtype=np.float64)
    omega_diag = jnp.array(np.diag(omega))

    dose_times = jnp.array([d.time for d in doses], dtype=jnp.float32)
    dose_amounts = jnp.array([d.amount for d in doses], dtype=jnp.float32)
    dose_durations = jnp.array(
        [d.duration if d.duration > 0 else 0.01 for d in doses],
        dtype=jnp.float32,
    )

    obs_times = jnp.array([o.time for o in observations], dtype=jnp.float32)
    y_obs = jnp.array(
        [o.concentration for o in observations], dtype=jnp.float32,
    )

    # Error model parameters
    sigma_prop = float(model.error_model.sigma_prop)
    sigma_add = float(model.error_model.sigma_add)

    # Suppress JAX warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        kernel = _NUTS(
            _pk_model_numpyro,
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
            omega_diag=omega_diag,
            tv_cl=float(tv_params.CL),
            tv_v1=float(tv_params.V1),
            tv_q=float(tv_params.Q),
            tv_v2=float(tv_params.V2),
            dose_times=dose_times,
            dose_amounts=dose_amounts,
            dose_durations=dose_durations,
            obs_times=obs_times,
            sigma_prop=sigma_prop,
            sigma_add=sigma_add,
            y_obs=y_obs,
        )

    # Extract posterior samples
    samples = mcmc.get_samples(group_by_chain=True)

    n_params = omega.shape[0]
    param_names = ["eta_CL", "eta_V1", "eta_Q", "eta_V2"][:n_params]
    pk_names = ["CL", "V1", "Q", "V2"][:n_params]

    # Compute diagnostics
    rhat_dict: dict[str, float] = {}
    ess_dict: dict[str, float] = {}
    summary_dict: dict[str, dict[str, float]] = {}
    all_eta = []

    for i, (eta_name, pk_name) in enumerate(zip(param_names, pk_names)):
        if eta_name in samples:
            chains = np.array(samples[eta_name])  # (n_chains, n_samples)
            rhat_dict[pk_name] = _compute_rhat(chains)

            all_samples = chains.flatten()
            ess_dict[pk_name] = _compute_ess(all_samples)

            tv_val = getattr(tv_params, pk_name)
            pk_samples = tv_val * np.exp(all_samples)

            summary_dict[pk_name] = {
                "mean": float(np.mean(pk_samples)),
                "sd": float(np.std(pk_samples, ddof=1)),
                "median": float(np.median(pk_samples)),
                "ci95_lower": float(np.percentile(pk_samples, 2.5)),
                "ci95_upper": float(np.percentile(pk_samples, 97.5)),
                "eta_mean": float(np.mean(all_samples)),
                "eta_sd": float(np.std(all_samples, ddof=1)),
            }
            all_eta.append(all_samples)
        else:
            tv_val = getattr(tv_params, pk_name)
            rhat_dict[pk_name] = 1.0
            ess_dict[pk_name] = float(n_samples * n_chains)
            summary_dict[pk_name] = {
                "mean": tv_val, "sd": 0.0, "median": tv_val,
                "ci95_lower": tv_val, "ci95_upper": tv_val,
                "eta_mean": 0.0, "eta_sd": 0.0,
            }
            all_eta.append(np.zeros(n_samples * n_chains))

    posterior_eta = np.column_stack(all_eta)

    # Count divergences
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

    map_eta = np.array([
        summary_dict[pk_names[i]]["eta_mean"]
        for i in range(n_params)
    ])
    map_params = apply_iiv(tv_params, map_eta)

    posterior_params: dict[str, dict[str, float]] = {}
    for pk_name in pk_names:
        if pk_name in summary_dict:
            s = summary_dict[pk_name]
            posterior_params[pk_name] = {
                "value": s["mean"],
                "ci95_lower": s["ci95_lower"],
                "ci95_upper": s["ci95_upper"],
                "unit": "L/h" if pk_name in ("CL", "Q") else "L",
            }

    return MCMCResult(
        posterior_eta=posterior_eta,
        posterior_params=posterior_params,
        summary=summary_dict,
        rhat=rhat_dict,
        ess=ess_dict,
        n_samples=n_samples * n_chains,
        n_chains=n_chains,
        n_divergences=n_divergences,
        converged=converged,
        map_params=map_params,
    )
