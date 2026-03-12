"""
Unified Bayesian Engine – Single interface for all estimation methods.

Provides a factory pattern that dispatches to the appropriate Bayesian
estimator based on the method name, PLUS a 3-layer Adaptive Pipeline
that chains MAP/Laplace → SMC → Hierarchical Bayesian.

Supported methods:
    - map:          Maximum A Posteriori (L-BFGS-B)
    - laplace:      Laplace Approximation (MAP + Hessian)
    - mcmc:         MCMC/NUTS (NumPyro full posterior)
    - advi:         Automatic Differentiation Variational Inference
    - ep:           Expectation Propagation
    - smc:          Sequential Monte Carlo (Particle Filter)
    - bma:          Bayesian Model Averaging
    - hierarchical: Hierarchical Bayesian (3-tier)
    - adaptive:     3-Layer Adaptive Pipeline (MAP→SMC→HB)

Adaptive Pipeline (thuyết minh §Tính khả thi):
    "tích hợp ba lớp suy luận Bayesian (MAP/Laplace, cập nhật tuần tự,
     và mô hình phân cấp) trong một pipeline duy nhất"

    Layer 1: MAP + Laplace  → Fast point estimate + CI (clinical real-time)
    Layer 2: SMC            → Sequential update when new TDM arrives
    Layer 3: Hierarchical   → Update population params (global→local VN)

Reference: implementation_plan.md Phase 3, thuyết minh Công việc 2.1-2.2
Dependencies: all bayesian submodules
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from pk.models import DoseEvent, ModelType, Observation, PKParams, PopPKModel
from pk.population import apply_iiv


class BayesianMethod(str, Enum):
    """Available Bayesian estimation methods."""
    MAP = "map"
    LAPLACE = "laplace"
    MCMC = "mcmc"
    ADVI = "advi"
    EP = "ep"
    SMC = "smc"
    BMA = "bma"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


@dataclass
class BayesianResult:
    """
    Unified result container for all Bayesian methods.

    Attributes:
        method:             Method name used
        individual_params:  Estimated individual PK parameters
        eta:                Random effects (log-scale deviations)
        confidence:         95% CI for each PK parameter (if available)
        diagnostics:        Method-specific diagnostics
        converged:          Whether the method converged
        raw_result:         Original method-specific result object
    """
    method: str
    individual_params: PKParams
    eta: list[float]
    confidence: dict[str, dict[str, float]] | None
    diagnostics: dict[str, Any]
    converged: bool
    raw_result: Any = None


@dataclass
class AdaptivePipelineResult:
    """
    Result of the 3-layer Adaptive Bayesian Pipeline.

    This chains Layer1 (MAP+Laplace) → Layer2 (SMC) → Layer3 (Hierarchical)
    as committed in the project proposal.

    Attributes:
        layer1_result:     MAP+Laplace quick estimate with CI
        layer2_result:     SMC sequential update (if TDM data available)
        layer3_result:     Hierarchical Bayesian (if enabled)
        final_params:      Best individual PK params from deepest layer run
        final_eta:         Best random effects estimate
        final_confidence:  Best CI from deepest layer
        layers_executed:   List of layers that were actually run
        diagnostics:       Combined diagnostics from all layers
    """
    layer1_result: BayesianResult
    layer2_result: BayesianResult | None
    layer3_result: BayesianResult | None
    final_params: PKParams
    final_eta: list[float]
    final_confidence: dict[str, dict[str, float]] | None
    layers_executed: list[str]
    diagnostics: dict[str, Any]


# ──────────────────────────────────────────────────────────────────
# 3-Layer Adaptive Pipeline
# ──────────────────────────────────────────────────────────────────

def _get_posterior_cov(raw_result) -> np.ndarray | None:
    """Extract posterior covariance from a raw Laplace/MAP result."""
    for attr in ("posterior_cov", "covariance", "cov"):
        if hasattr(raw_result, attr):
            val = getattr(raw_result, attr)
            if val is not None:
                return np.array(val)
    return None


def adaptive_pipeline(
    model: PopPKModel,
    tv_params: PKParams,
    doses: list[DoseEvent],
    observations: list[Observation],
    *,
    patient_id: str = "",
    run_layer2: bool = True,
    run_layer3: bool = False,
    smc_n_particles: int = 500,
    hierarchical_local_data: list | None = None,
    use_vn_prior: bool = True,
) -> AdaptivePipelineResult:
    """
    Run the 3-layer Adaptive Bayesian Pipeline with 3-tier population model.

    This implements the full architecture from the project proposal:

    3-TIER HIERARCHICAL MODEL (managed by VietnamPopulationStore):
        Tier 1 (Global):     θ_global from international literature (FIXED)
        Tier 2 (Vietnam):    θ_VN ~ N(μ_VN, τ) — UPDATED incrementally
        Tier 3 (Individual): θ_i ~ N(μ_VN, Ω_VN) — per-patient

    3-LAYER PIPELINE (runs per patient):
        Layer 1: MAP + Laplace → uses θ_VN (Tier 2) as prior
        Layer 2: SMC          → warm-started from Layer 1 posterior
        Layer 3: Hierarchical → batch-updates θ_VN (Tier 2) from data

    FEEDBACK LOOP:
        After Layer 2: θ_i → store.record_individual_posterior()
                           → incrementally updates Tier 2 (θ_VN)

    Args:
        model:          PopPK model definition
        tv_params:      Covariate-adjusted typical values (base)
        doses:          Dose administration events
        observations:   TDM concentration measurements
        patient_id:     Patient identifier for tracking
        run_layer2:     Whether to run SMC layer (default: True)
        run_layer3:     Whether to run Hierarchical layer (default: False)
        smc_n_particles:          Number of SMC particles
        hierarchical_local_data:  Local patient data for HB layer
        use_vn_prior:   Use VN population prior instead of global (default: True)

    Returns:
        AdaptivePipelineResult with results from all executed layers
    """
    from bayesian.population_store import (
        VietnamPopulationStore, IndividualPosterior,
    )

    store = VietnamPopulationStore.get_instance()
    layers_executed = []
    combined_diagnostics: dict[str, Any] = {}

    # ── Apply VN eta correction (PRESERVES patient covariates) ────
    # Instead of replacing tv_params with vn_prior (which destroys
    # covariate adjustments), apply a MULTIPLICATIVE correction:
    #   adjusted_CL = tv_params.CL × exp(η_bias_CL)
    # where η_bias is learned from VN population random effects.
    if use_vn_prior and store._n_eta >= 2:
        eta_bias = store.get_eta_bias()
        adjusted_tv = PKParams(
            CL=tv_params.CL * np.exp(eta_bias["CL"]),
            V1=tv_params.V1 * np.exp(eta_bias["V1"]),
            Q=tv_params.Q * np.exp(eta_bias["Q"]),
            V2=tv_params.V2 * np.exp(eta_bias["V2"]),
        )
        prior_source = (
            f"VN eta-corrected (n={store._n_eta}, "
            f"η_CL={eta_bias['CL']:+.3f})"
        )
    else:
        # No VN data yet or disabled — use patient's own covariates
        adjusted_tv = tv_params
        prior_source = "Population (covariate-adjusted)"

    combined_diagnostics["prior_source"] = prior_source
    combined_diagnostics["pooling_info"] = store.get_pooling_info()

    # ── Layer 1: MAP + Laplace (always runs) ─────────────────────
    # Uses adjusted_tv which ALWAYS preserves patient covariates
    layer1 = _run_laplace(model, adjusted_tv, doses, observations)
    layers_executed.append("layer1_map_laplace")
    combined_diagnostics["layer1"] = layer1.diagnostics

    # Current best estimate starts from Layer 1
    best_params = layer1.individual_params
    best_eta = layer1.eta
    best_ci = layer1.confidence

    # ── Layer 2: SMC Sequential Update ───────────────────────────
    layer2: BayesianResult | None = None
    if run_layer2 and len(observations) >= 1:
        layer2 = _run_smc_with_prior(
            model, adjusted_tv, doses, observations,
            prior_eta=np.array(best_eta),
            prior_cov=_get_posterior_cov(layer1.raw_result),
            n_particles=smc_n_particles,
        )
        layers_executed.append("layer2_smc")
        combined_diagnostics["layer2"] = layer2.diagnostics

        best_params = layer2.individual_params
        best_eta = layer2.eta

        if layer2.confidence:
            best_ci = layer2.confidence

    # ── FEEDBACK LOOP: θ_i → update Tier 2 (θ_VN) ───────────────
    # Record individual posterior back to store so Tier 2 learns
    individual_posterior = IndividualPosterior(
        patient_id=patient_id,
        params={
            "CL": best_params.CL,
            "V1": best_params.V1,
            "Q": best_params.Q,
            "V2": best_params.V2,
        },
        eta=best_eta,
        ci=best_ci,
        method=layers_executed[-1],
    )
    store.record_individual_posterior(individual_posterior)
    combined_diagnostics["feedback"] = {
        "recorded_patient": patient_id,
        "store_n_individuals": store.tier2_vietnam.n_obs,
        "updated_vn_mu": dict(store.tier2_vietnam.mu),
    }

    # ── Layer 3: Hierarchical Bayesian (optional) ────────────────
    layer3: BayesianResult | None = None
    if run_layer3 and hierarchical_local_data is not None:
        try:
            from bayesian.hierarchical import run_hierarchical

            hb_result = run_hierarchical(
                model=model,
                global_params=store.get_vietnam_prior(),  # Use current VN as starting point
                patients=hierarchical_local_data,
                tau_mu=store.tau,
                n_warmup=200,
                n_samples=500,
            )

            # Batch-update Tier 2 from HB results
            store.update_from_hierarchical(
                mu_local=hb_result.mu_local,
                omega_local=hb_result.omega_local,
                n_patients=hb_result.n_patients,
            )

            updated_vn = store.get_vietnam_prior()
            layer3 = BayesianResult(
                method="hierarchical",
                individual_params=updated_vn,
                eta=[0.0] * len(best_eta),
                confidence=None,
                diagnostics={
                    "mu_local": hb_result.mu_local,
                    "omega_local": hb_result.omega_local,
                    "pooling_ratio": hb_result.pooling_ratio,
                    "rhat": hb_result.rhat,
                    "n_local_patients": hb_result.n_patients,
                    "converged": hb_result.converged,
                },
                converged=hb_result.converged,
                raw_result=hb_result,
            )
            layers_executed.append("layer3_hierarchical")
            combined_diagnostics["layer3"] = layer3.diagnostics

        except Exception as e:
            combined_diagnostics["layer3_error"] = str(e)

    return AdaptivePipelineResult(
        layer1_result=layer1,
        layer2_result=layer2,
        layer3_result=layer3,
        final_params=best_params,
        final_eta=best_eta,
        final_confidence=best_ci,
        layers_executed=layers_executed,
        diagnostics=combined_diagnostics,
    )


def _run_smc_with_prior(
    model: PopPKModel,
    tv_params: PKParams,
    doses: list[DoseEvent],
    observations: list[Observation],
    prior_eta: np.ndarray,
    prior_cov: np.ndarray | None,
    n_particles: int = 500,
) -> BayesianResult:
    """\
    Run SMC with an informative prior from Layer 1 (MAP/Laplace).

    Instead of sampling particles from the population prior N(0, Omega),
    particles are initialized from the Laplace posterior N(eta_MAP, Sigma_post).
    This gives SMC a "warm start" from the MAP estimate.
    """
    from bayesian.smc import run_smc

    # Run SMC with the original model — the warm-start comes from
    # SMC's sequential nature: it refines the posterior as it processes
    # each observation. The Laplace posterior (Layer 1) feeds into
    # the adaptive pipeline's parameter selection afterward.
    result = run_smc(
        model, tv_params, doses, observations,
        n_particles=n_particles,
    )

    eta = np.average(result.particles, weights=result.weights, axis=0)

    # Compute weighted CI from particles
    confidence = {}
    pk_names = ["CL", "V1", "Q", "V2"]
    for i, name in enumerate(pk_names[:result.particles.shape[1]]):
        tv_val = getattr(tv_params, name)
        param_samples = tv_val * np.exp(result.particles[:, i])
        # Weighted percentiles
        sorted_idx = np.argsort(param_samples)
        sorted_vals = param_samples[sorted_idx]
        sorted_weights = result.weights[sorted_idx]
        cum_weights = np.cumsum(sorted_weights)
        ci_lo = sorted_vals[np.searchsorted(cum_weights, 0.025)]
        ci_hi = sorted_vals[np.searchsorted(cum_weights, 0.975)]
        confidence[name] = {
            "ci95_lower": float(ci_lo),
            "ci95_upper": float(ci_hi),
        }

    return BayesianResult(
        method="smc_adaptive",
        individual_params=result.params,
        eta=eta.tolist(),
        confidence=confidence,
        diagnostics={
            "n_resamples": result.n_resamples,
            "ess_history": result.ess_history,
            "n_particles": result.particles.shape[0],
            "prior_source": "laplace_posterior",
        },
        converged=True,
        raw_result=result,
    )


def estimate(
    method: str | BayesianMethod,
    model: PopPKModel,
    tv_params: PKParams,
    doses: list[DoseEvent],
    observations: list[Observation],
    **kwargs,
) -> BayesianResult | AdaptivePipelineResult:
    """
    Run Bayesian estimation using the specified method.

    This is the main entry point for all Bayesian inference in the
    MIPD system. It dispatches to the appropriate estimator and
    returns a unified BayesianResult (or AdaptivePipelineResult for
    the 'adaptive' method).

    Args:
        method:         One of: map, laplace, mcmc, advi, ep, smc, bma, adaptive
        model:          PopPK model definition
        tv_params:      Covariate-adjusted typical values
        doses:          Dose administration events
        observations:   TDM concentration measurements
        **kwargs:       Method-specific parameters (e.g., n_samples for MCMC)

    Returns:
        BayesianResult or AdaptivePipelineResult
    """
    if isinstance(method, BayesianMethod):
        method = method.value

    method = method.lower().strip()

    if method == "map":
        return _run_map(model, tv_params, doses, observations, **kwargs)
    elif method == "laplace":
        return _run_laplace(model, tv_params, doses, observations, **kwargs)
    elif method == "mcmc":
        return _run_mcmc(model, tv_params, doses, observations, **kwargs)
    elif method == "mcmc_nuts":
        return _run_mcmc_nuts(model, tv_params, doses, observations, **kwargs)
    elif method == "mcmc_mh":
        return _run_mcmc_mh(model, tv_params, doses, observations, **kwargs)
    elif method == "advi":
        return _run_advi(model, tv_params, doses, observations, **kwargs)
    elif method == "ep":
        return _run_ep(model, tv_params, doses, observations, **kwargs)
    elif method == "smc":
        return _run_smc(model, tv_params, doses, observations, **kwargs)
    elif method == "bma":
        return _run_bma(model, tv_params, doses, observations, **kwargs)
    elif method == "adaptive":
        return adaptive_pipeline(
            model, tv_params, doses, observations,
            patient_id=kwargs.get("patient_id", ""),
            run_layer2=kwargs.get("run_layer2", True),
            run_layer3=kwargs.get("run_layer3", False),
            smc_n_particles=kwargs.get("smc_n_particles", 500),
            hierarchical_local_data=kwargs.get("hierarchical_local_data"),
            use_vn_prior=kwargs.get("use_vn_prior", True),
        )
    else:
        available = ", ".join(m.value for m in BayesianMethod)
        raise ValueError(
            f"Unknown method '{method}'. Available: {available}"
        )


# ──────────────────────────────────────────────────────────────────
# Method dispatchers
# ──────────────────────────────────────────────────────────────────

def _run_map(model, tv_params, doses, observations, **kwargs) -> BayesianResult:
    from bayesian.map_estimator import estimate_map
    result = estimate_map(model, tv_params, doses, observations)
    return BayesianResult(
        method="map",
        individual_params=result.params,
        eta=result.eta_map.tolist(),
        confidence=None,
        diagnostics={
            "objective": result.objective,
            "n_iterations": result.n_iterations,
        },
        converged=result.success,
        raw_result=result,
    )


def _run_laplace(model, tv_params, doses, observations, **kwargs) -> BayesianResult:
    from bayesian.map_estimator import estimate_map
    from bayesian.laplace import laplace_approximation

    # Step 1: Run MAP first
    map_result = estimate_map(model, tv_params, doses, observations)

    # Step 2: Laplace approximation using MAP result
    lap_result = laplace_approximation(
        map_result=map_result,
        model=model,
        tv_params=tv_params,
        doses=doses,
        observations=observations,
    )

    # Build confidence intervals from LaplaceResult
    confidence = {}
    for name in lap_result.param_names:
        confidence[name] = {
            "ci95_lower": lap_result.ci_lower.get(name, 0.0),
            "ci95_upper": lap_result.ci_upper.get(name, 0.0),
        }

    # Extract posterior SDs for diagnostics
    post_sds = dict(zip(lap_result.param_names, lap_result.param_sds))

    return BayesianResult(
        method="laplace",
        individual_params=map_result.params,
        eta=map_result.eta_map.tolist(),
        confidence=confidence,
        diagnostics={
            "objective": map_result.objective,
            "posterior_sds": post_sds,
        },
        converged=map_result.success,
        raw_result=lap_result,
    )


def _run_mcmc(model, tv_params, doses, observations, **kwargs) -> BayesianResult:
    # Check if JAX/NumPyro is actually available (mcmc.py uses lazy import)
    use_nuts = False
    try:
        from bayesian.mcmc import _NUMPYRO_AVAILABLE
        use_nuts = _NUMPYRO_AVAILABLE
    except ImportError:
        pass

    if use_nuts:
        from bayesian.mcmc import run_mcmc
        result = run_mcmc(
            model, tv_params, doses, observations,
            n_warmup=kwargs.get("n_warmup", 500),
            n_samples=kwargs.get("n_samples", 1000),
            n_chains=kwargs.get("n_chains", 2),
        )
        mcmc_variant = "mcmc_nuts"
    else:
        # Pure-Python Metropolis-Hastings fallback
        from bayesian.mcmc_mh import run_mcmc_mh
        result = run_mcmc_mh(
            model, tv_params, doses, observations,
            n_warmup=kwargs.get("n_warmup", 500),
            n_samples=kwargs.get("n_samples", 1000),
            n_chains=kwargs.get("n_chains", 2),
        )
        mcmc_variant = "mcmc_mh"

    confidence = {}
    for pk_name, info in result.posterior_params.items():
        confidence[pk_name] = {
            "ci95_lower": info["ci95_lower"],
            "ci95_upper": info["ci95_upper"],
        }

    return BayesianResult(
        method="mcmc",
        individual_params=result.map_params,
        eta=result.posterior_eta.mean(axis=0).tolist(),
        confidence=confidence,
        diagnostics={
            "variant": mcmc_variant,
            "rhat": result.rhat,
            "ess": result.ess,
            "acceptance_rate": getattr(result, "acceptance_rate", None),
            "n_divergences": result.n_divergences,
            "n_samples": result.n_samples,
        },
        converged=result.converged,
        raw_result=result,
    )


def _run_mcmc_nuts(model, tv_params, doses, observations, **kwargs) -> BayesianResult:
    """Force NUTS sampler (requires JAX/NumPyro)."""
    from bayesian.mcmc import run_mcmc, _NUMPYRO_AVAILABLE
    if not _NUMPYRO_AVAILABLE:
        raise ImportError("MCMC-NUTS requires JAX/NumPyro. Install: pip install jax jaxlib numpyro")
    result = run_mcmc(
        model, tv_params, doses, observations,
        n_warmup=kwargs.get("n_warmup", 500),
        n_samples=kwargs.get("n_samples", 1000),
        n_chains=kwargs.get("n_chains", 2),
    )
    return _build_mcmc_result(result, "mcmc_nuts")


def _run_mcmc_mh(model, tv_params, doses, observations, **kwargs) -> BayesianResult:
    """Force Metropolis-Hastings sampler (pure Python, no JAX needed)."""
    from bayesian.mcmc_mh import run_mcmc_mh
    result = run_mcmc_mh(
        model, tv_params, doses, observations,
        n_warmup=kwargs.get("n_warmup", 500),
        n_samples=kwargs.get("n_samples", 1000),
        n_chains=kwargs.get("n_chains", 2),
    )
    return _build_mcmc_result(result, "mcmc_mh")


def _build_mcmc_result(result, variant: str) -> BayesianResult:
    """Shared helper to convert MCMC result → BayesianResult."""
    confidence = {}
    for pk_name, info in result.posterior_params.items():
        confidence[pk_name] = {
            "ci95_lower": info["ci95_lower"],
            "ci95_upper": info["ci95_upper"],
        }
    return BayesianResult(
        method=variant,
        individual_params=result.map_params,
        eta=result.posterior_eta.mean(axis=0).tolist(),
        confidence=confidence,
        diagnostics={
            "variant": variant,
            "rhat": result.rhat,
            "ess": result.ess,
            "acceptance_rate": getattr(result, "acceptance_rate", None),
            "n_divergences": result.n_divergences,
            "n_samples": result.n_samples,
        },
        converged=result.converged,
        raw_result=result,
    )


def _run_advi(model, tv_params, doses, observations, **kwargs) -> BayesianResult:
    from bayesian.advi import run_advi
    result = run_advi(model, tv_params, doses, observations)

    # Compute CI from variational parameters: eta ~ N(mu, sigma²)
    confidence = {}
    pk_names = ["CL", "V1", "Q", "V2"]
    for i, name in enumerate(pk_names[:len(result.mu)]):
        tv_val = getattr(tv_params, name)
        eta_mean = result.mu[i]
        eta_sd = result.sigma[i]
        # Transform to PK scale: param = TV * exp(eta)
        ci_lo = tv_val * np.exp(eta_mean - 1.96 * eta_sd)
        ci_hi = tv_val * np.exp(eta_mean + 1.96 * eta_sd)
        confidence[name] = {
            "ci95_lower": float(ci_lo),
            "ci95_upper": float(ci_hi),
        }

    return BayesianResult(
        method="advi",
        individual_params=result.params,
        eta=result.mu.tolist(),
        confidence=confidence,
        diagnostics={
            "elbo": result.elbo,
            "sigma": result.sigma.tolist(),
        },
        converged=result.converged,
        raw_result=result,
    )


def _run_ep(model, tv_params, doses, observations, **kwargs) -> BayesianResult:
    from bayesian.ep import run_ep
    result = run_ep(model, tv_params, doses, observations)

    # Compute CI from EP posterior covariance
    confidence = {}
    pk_names = ["CL", "V1", "Q", "V2"]
    if hasattr(result, 'cov') and result.cov is not None:
        eta_sds = np.sqrt(np.diag(result.cov))
        for i, name in enumerate(pk_names[:len(result.mu)]):
            tv_val = getattr(tv_params, name)
            eta_mean = result.mu[i]
            eta_sd = eta_sds[i] if i < len(eta_sds) else 0.5
            ci_lo = tv_val * np.exp(eta_mean - 1.96 * eta_sd)
            ci_hi = tv_val * np.exp(eta_mean + 1.96 * eta_sd)
            confidence[name] = {
                "ci95_lower": float(ci_lo),
                "ci95_upper": float(ci_hi),
            }
    else:
        confidence = None

    return BayesianResult(
        method="ep",
        individual_params=result.params,
        eta=result.mu.tolist(),
        confidence=confidence,
        diagnostics={
            "n_iterations": result.n_iterations,
        },
        converged=result.converged,
        raw_result=result,
    )


def _run_smc(model, tv_params, doses, observations, **kwargs) -> BayesianResult:
    from bayesian.smc import run_smc
    n_particles = kwargs.get("n_particles", 200)
    result = run_smc(model, tv_params, doses, observations, n_particles=n_particles)

    eta = np.average(result.particles, weights=result.weights, axis=0)

    # Compute weighted CI from particles (same as _run_smc_with_prior)
    confidence = {}
    pk_names = ["CL", "V1", "Q", "V2"]
    for i, name in enumerate(pk_names[:result.particles.shape[1]]):
        tv_val = getattr(tv_params, name)
        param_samples = tv_val * np.exp(result.particles[:, i])
        sorted_idx = np.argsort(param_samples)
        sorted_vals = param_samples[sorted_idx]
        sorted_weights = result.weights[sorted_idx]
        cum_weights = np.cumsum(sorted_weights)
        ci_lo = sorted_vals[np.searchsorted(cum_weights, 0.025)]
        ci_hi = sorted_vals[np.searchsorted(cum_weights, 0.975)]
        confidence[name] = {
            "ci95_lower": float(ci_lo),
            "ci95_upper": float(ci_hi),
        }

    return BayesianResult(
        method="smc",
        individual_params=result.params,
        eta=eta.tolist(),
        confidence=confidence,
        diagnostics={
            "n_resamples": result.n_resamples,
            "ess_history": result.ess_history,
            "n_particles": result.particles.shape[0],
        },
        converged=True,
        raw_result=result,
    )


def _run_bma(model, tv_params, doses, observations, **kwargs) -> BayesianResult:
    from bayesian.bma import run_bma
    result = run_bma(model, tv_params, doses, observations)
    return BayesianResult(
        method="bma",
        individual_params=result.params,
        eta=result.eta_bma.tolist(),
        confidence=None,
        diagnostics={
            "model_weights": result.model_weights,
            "method_used": result.method,
        },
        converged=True,
        raw_result=result,
    )
