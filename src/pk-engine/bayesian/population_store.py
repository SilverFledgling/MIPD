"""
Vietnam Population Store – 3-Tier Hierarchical Parameter Management.

Manages the relationship between the 3-tier hierarchical model
(Global → Vietnam → Individual) and the 3-layer adaptive pipeline
(MAP/Laplace → SMC → Hierarchical).

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │   MÔ HÌNH PHÂN CẤP 3 TẦNG                          │
    │                                                     │
    │   Tier 1 (Global):  θ_global from literature        │
    │       ↓ hyper-prior τ                               │
    │   Tier 2 (Vietnam): θ_VN ~ N(μ_VN, τ)              │
    │       ↓ population prior    ↑ feedback loop         │
    │   Tier 3 (Individual): θ_i ~ N(μ_VN, Ω_VN)        │
    └─────────────────────────────────────────────────────┘

    - VietnamPopulationStore HOLDS Tier 1 + Tier 2 parameters
    - Tier 2 (θ_VN) is UPDATED incrementally as individual posteriors
      are fed back from the adaptive pipeline
    - Tier 1 (θ_global) is FIXED from international literature

Key concept (from thuyết minh Công việc 2.2):
    "Tích hợp Bayesian mô hình đa tầng và cập nhật liên tục
     bằng dữ liệu nội địa"

    → This module IS the "cập nhật liên tục" mechanism.

Reference:
    - Gelman et al. (2013), BDA, Ch. 5 (partial pooling)
    - NAFOSTED proposal: Công việc 2.2
    - mathematical_reference.md §4.11

Dependencies: numpy, pk.models
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pk.models import PKParams


# ──────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────

@dataclass
class TierParams:
    """
    Parameters for one tier in the hierarchical model.

    Attributes:
        mu:      Mean values {CL, V1, Q, V2}
        omega:   Variance (IIV) for each parameter
        n_obs:   Number of observations that contributed to this estimate
        source:  Description of where these params come from
    """
    mu: dict[str, float]
    omega: dict[str, float]
    n_obs: int = 0
    source: str = ""


@dataclass
class IndividualPosterior:
    """
    Posterior estimate from one patient (output of Layer 1 or 2).

    Used as feedback to update Tier 2 (Vietnam population).
    """
    patient_id: str
    params: dict[str, float]    # {CL: 3.21, V1: 26.5, Q: 4.65, V2: 37.0}
    eta: list[float]            # Random effects
    ci: dict[str, dict[str, float]] | None = None  # 95% CI per param
    method: str = ""            # Which layer produced this


# ──────────────────────────────────────────────────────────────────
# Vietnam Population Store
# ──────────────────────────────────────────────────────────────────

class VietnamPopulationStore:
    """
    Persistent store for 3-tier hierarchical population parameters.

    Manages:
        - Tier 1 (Global): Fixed international prior (Goti 2018, Thomson 2009)
        - Tier 2 (Vietnam): Updated incrementally from individual posteriors
        - Feedback loop: individual θ_i → update μ_VN, Ω_VN

    Thread-safe for concurrent access from API endpoints.

    Usage:
        store = VietnamPopulationStore.get_instance()

        # Get VN prior for adaptive pipeline Layer 1
        vn_params = store.get_vietnam_prior()

        # After pipeline finishes, feed back individual posterior
        store.record_individual_posterior(IndividualPosterior(...))
    """

    _instance: VietnamPopulationStore | None = None
    _lock = threading.Lock()

    def __init__(self, data_dir: str | Path | None = None):
        """
        Initialize store with global priors.

        Args:
            data_dir: Directory to persist state (JSON). If None, in-memory only.
        """
        self._data_dir = Path(data_dir) if data_dir else None
        self._rw_lock = threading.RLock()

        # ── Tier 1: Global priors (FIXED, from international literature) ──
        # Omega values MUST match VANCOMYCIN_VN in pk/population.py
        self.tier1_global = TierParams(
            mu={
                "CL": 4.50,   # L/h — Goti 2018 + Thomson 2009 pooled
                "V1": 30.0,   # L   — Central volume
                "Q":  4.50,   # L/h — Inter-compartmental clearance
                "V2": 40.0,   # L   — Peripheral volume
            },
            omega={
                "CL": 0.150,  # ~41% CV — matches VANCOMYCIN_VN model
                "V1": 0.100,  # ~33% CV
                "Q":  0.200,  # ~47% CV
                "V2": 0.150,  # ~41% CV
            },
            n_obs=0,
            source="International literature (Goti 2018, Thomson 2009)",
        )

        # ── Tier 2: Vietnam population (UPDATABLE) ──
        # Initialize from global — will be updated as VN data accumulates
        self.tier2_vietnam = TierParams(
            mu=dict(self.tier1_global.mu),       # Start = global
            omega=dict(self.tier1_global.omega),  # Start = global
            n_obs=0,
            source="Initialized from global (no VN data yet)",
        )

        # ── Individual posteriors history (for batch re-estimation) ──
        self._individual_history: list[IndividualPosterior] = []

        # ── Eta tracking (for covariate-preserving correction) ──
        # mu_eta: mean random effect per parameter (should be ~0 if model unbiased)
        # omega_eta: variance of random effects (= empirical IIV)
        self._mu_eta: dict[str, float] = {"CL": 0.0, "V1": 0.0, "Q": 0.0, "V2": 0.0}
        self._omega_eta: dict[str, float] = dict(self.tier1_global.omega)
        self._n_eta: int = 0

        # ── Tau: shrinkage parameter (how much VN can deviate from global) ──
        self.tau = 0.3  # Prior std on log scale

        # Load persisted state if available
        if self._data_dir:
            self._load_state()

    @classmethod
    def get_instance(cls, data_dir: str | Path | None = None) -> VietnamPopulationStore:
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(data_dir=data_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    # ── Public API ───────────────────────────────────────────────

    def get_vietnam_prior(self) -> PKParams:
        """
        Get current Vietnam population parameters as PKParams.

        This is used as the PRIOR for the adaptive pipeline Layer 1.
        When VN data is sparse, this ≈ global. When VN data is rich,
        this reflects Vietnamese-specific PK characteristics.

        Returns:
            PKParams with μ_VN values
        """
        with self._rw_lock:
            return PKParams(
                CL=self.tier2_vietnam.mu["CL"],
                V1=self.tier2_vietnam.mu["V1"],
                Q=self.tier2_vietnam.mu["Q"],
                V2=self.tier2_vietnam.mu["V2"],
            )

    def get_eta_bias(self) -> dict[str, float]:
        """
        Get mean eta (random effect) per parameter from VN population.

        Used for COVARIATE-PRESERVING correction:
            adjusted_CL = tv_params.CL × exp(eta_bias["CL"])

        If VN patients have lower CL than predicted by global model:
            eta_bias["CL"] < 0 → adjusted_CL < tv_params.CL

        Returns:
            Dict {"CL": float, "V1": float, "Q": float, "V2": float}
            Values are 0.0 when no VN data available.
        """
        with self._rw_lock:
            if self._n_eta < 2:
                return {"CL": 0.0, "V1": 0.0, "Q": 0.0, "V2": 0.0}
            return dict(self._mu_eta)

    def get_vietnam_omega(self) -> list[list[float]]:
        """
        Get current Vietnam population omega (IIV) as matrix.

        Used to set the prior covariance for Bayesian estimation.

        Returns:
            4x4 diagonal omega matrix
        """
        with self._rw_lock:
            omega = self.tier2_vietnam.omega
            n = 4
            matrix = [[0.0] * n for _ in range(n)]
            for i, key in enumerate(["CL", "V1", "Q", "V2"]):
                matrix[i][i] = omega.get(key, 0.1)
            return matrix

    def get_pooling_info(self) -> dict[str, Any]:
        """
        Get diagnostic info about pooling (how much VN differs from global).

        Returns:
            Dict with per-parameter pooling ratios and observation counts.
        """
        with self._rw_lock:
            pooling = {}
            for key in ["CL", "V1", "Q", "V2"]:
                g = self.tier1_global.mu[key]
                v = self.tier2_vietnam.mu[key]
                deviation = abs(v - g) / g if g > 0 else 0
                pooling[key] = {
                    "global_mu": g,
                    "vietnam_mu": v,
                    "deviation_pct": round(deviation * 100, 2),
                    "pooling_ratio": round(min(deviation / self.tau, 1.0), 3),
                }
            pooling["n_individuals"] = len(self._individual_history)
            pooling["n_observations"] = self.tier2_vietnam.n_obs
            pooling["source"] = self.tier2_vietnam.source
            return pooling

    def record_individual_posterior(self, posterior: IndividualPosterior):
        """
        Record an individual patient's posterior → triggers Tier 2 update.

        This implements the FEEDBACK LOOP:
            θ_i (from pipeline Layer 2) → update μ_VN, Ω_VN (Tier 2)

        Uses incremental Bayesian updating (conjugate Normal-Normal):
            μ_new = (n·μ_old + θ_i) / (n + 1)
            Ω_new = updated weighted variance

        Args:
            posterior: Individual posterior from adaptive pipeline
        """
        with self._rw_lock:
            self._individual_history.append(posterior)
            self._update_tier2_incremental(posterior)
            self._persist_state()

    def update_from_hierarchical(
        self,
        mu_local: dict[str, dict[str, float]],
        omega_local: dict[str, dict[str, float]],
        n_patients: int,
    ):
        """
        Batch update Tier 2 from a full Hierarchical Bayesian run.

        This is called after Layer 3 (HB) finishes with multi-patient
        data. The posterior μ_local and Ω_local REPLACE the current
        Tier 2 estimates (since HB uses all data simultaneously).

        Args:
            mu_local:    Posterior local means from HierarchicalResult
            omega_local: Posterior local omegas from HierarchicalResult
            n_patients:  Number of patients used in HB run
        """
        with self._rw_lock:
            for key in ["CL", "V1", "Q", "V2"]:
                if key in mu_local:
                    self.tier2_vietnam.mu[key] = mu_local[key].get(
                        "mean", self.tier2_vietnam.mu[key]
                    )
                if key in omega_local:
                    self.tier2_vietnam.omega[key] = omega_local[key].get(
                        "mean", self.tier2_vietnam.omega[key]
                    )

            self.tier2_vietnam.n_obs = n_patients
            self.tier2_vietnam.source = (
                f"Hierarchical Bayesian (n={n_patients} VN patients)"
            )
            self._persist_state()

    # ── Private methods ──────────────────────────────────────────

    def _update_tier2_incremental(self, posterior: IndividualPosterior):
        """
        SAEM-inspired Robbins-Monro update of Tier 2 — ETA-BASED.

        Tracks eta (random effects) instead of absolute parameter values,
        separating population learning from covariate adjustments.

        Eta update (Robbins-Monro):
            μ_η_{n+1} = μ_η_n + γ_n × (η_i − μ_η_n)
            ω_η_{n+1} = ω_η_n + γ_n × ((η_i − μ_η_{n+1})² − ω_η_n)

        Also maintains absolute mu/omega for backward compatibility.

        Reference: Delyon, Lavielle & Moulines (1999), SAEM algorithm.
        """
        n = self.tier2_vietnam.n_obs + 1

        # SAEM gain sequence: γ = 1/n^α, α = 0.6
        SA_ALPHA = 0.6
        gamma = 1.0 / (n ** SA_ALPHA)

        # ── Update ETA tracking (primary) ──
        # eta = log-scale random effects, covariate-independent
        pk_keys = ["CL", "V1", "Q", "V2"]
        if posterior.eta is not None and len(posterior.eta) >= len(pk_keys):
            self._n_eta += 1
            gamma_eta = 1.0 / (self._n_eta ** SA_ALPHA)

            for i, key in enumerate(pk_keys):
                if i >= len(posterior.eta):
                    break
                eta_i = posterior.eta[i]
                mu_eta_old = self._mu_eta[key]
                omega_eta_old = max(self._omega_eta[key], 1e-6)

                # Robbins-Monro mean update for eta
                self._mu_eta[key] = mu_eta_old + gamma_eta * (eta_i - mu_eta_old)

                # SA variance update for eta
                if self._n_eta >= 2:
                    sq_dev = (eta_i - self._mu_eta[key]) ** 2
                    new_omega = omega_eta_old + gamma_eta * (sq_dev - omega_eta_old)
                    self._omega_eta[key] = max(new_omega, 1e-6)

        # ── Update absolute mu/omega (backward compat) ──
        for key in pk_keys:
            if key not in posterior.params:
                continue

            theta_i = posterior.params[key]
            mu_old = self.tier2_vietnam.mu[key]

            # Precision weighting
            omega_old = max(self.tier2_vietnam.omega[key], 1e-6)
            w_i = 1.0
            if posterior.ci and key in posterior.ci:
                ci = posterior.ci[key]
                ci_width = ci.get("ci95_upper", theta_i) - ci.get("ci95_lower", theta_i)
                if ci_width > 0:
                    sigma2_i = (ci_width / (2 * 1.96)) ** 2
                    w_i = omega_old / (omega_old + sigma2_i)

            mu_new = mu_old + gamma * w_i * (theta_i - mu_old)
            g = self.tier1_global.mu[key]
            mu_new = max(g * 0.3, min(mu_new, g * 3.0))
            self.tier2_vietnam.mu[key] = mu_new

            if n >= 2:
                sq_dev = (theta_i - mu_new) ** 2
                new_omega = omega_old + gamma * (sq_dev - omega_old)
                self.tier2_vietnam.omega[key] = max(new_omega, 1e-6)

        self.tier2_vietnam.n_obs = n
        self.tier2_vietnam.source = (
            f"SAEM-SA update (n={n}, γ={gamma:.4f}, η_bias_CL={self._mu_eta['CL']:+.3f})"
        )

    def _persist_state(self):
        """Save current state to JSON file."""
        if self._data_dir is None:
            return

        self._data_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "tier1_global": {
                "mu": self.tier1_global.mu,
                "omega": self.tier1_global.omega,
                "source": self.tier1_global.source,
            },
            "tier2_vietnam": {
                "mu": self.tier2_vietnam.mu,
                "omega": self.tier2_vietnam.omega,
                "n_obs": self.tier2_vietnam.n_obs,
                "source": self.tier2_vietnam.source,
            },
            "tau": self.tau,
            "n_individuals": len(self._individual_history),
        }
        filepath = self._data_dir / "population_state.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _load_state(self):
        """Load persisted state from JSON file."""
        if self._data_dir is None:
            return

        filepath = self._data_dir / "population_state.json"
        if not filepath.exists():
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)

            if "tier2_vietnam" in state:
                t2 = state["tier2_vietnam"]
                self.tier2_vietnam.mu = t2.get("mu", self.tier2_vietnam.mu)
                self.tier2_vietnam.omega = t2.get("omega", self.tier2_vietnam.omega)
                self.tier2_vietnam.n_obs = t2.get("n_obs", 0)
                self.tier2_vietnam.source = t2.get("source", "Loaded from disk")

            if "tau" in state:
                self.tau = state["tau"]

        except (json.JSONDecodeError, KeyError) as e:
            pass  # Silently ignore corrupt state files
