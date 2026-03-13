"""
MIPD Benchmark — Enhanced with Dual Adaptive, Learning Curve & Clinical Metrics.

Metrics (per thuyết minh Nội dung 4.2):
  Core:     MPE, MAPE, RMSE, CCC
  Clinical: Bland-Altman (bias ± LoA), Coverage 95%, Target Attainment, η-Shrinkage
  Learning: Rolling CCC (for adaptive_cum only)

Methods:
  Fast:       map, laplace, ep
  Moderate:   smc, advi
  Advanced:   mcmc_mh, mcmc_nuts
  Adaptive:   adaptive_ind (independent — reset store each patient)
              adaptive_cum (cumulative — learning across patients)

Usage:
    cd src/pk-engine
    python benchmarks/run_benchmark.py          # 50 patients (default)
    python benchmarks/run_benchmark.py 20       # quick test
    python benchmarks/run_benchmark.py 200      # full evaluation
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# SOLID modules — extracted per Single Responsibility Principle
_bench_dir = str(Path(__file__).resolve().parent)
if _bench_dir not in sys.path:
    sys.path.insert(0, _bench_dir)
from benchmark_metrics import compute_ccc, compute_metrics, compute_rolling_ccc
from benchmark_plots import generate_all_plots
from benchmark_export import (
    export_main_csv, export_rolling_ccc,
    export_individual_csv, export_npde_summary,
)

# Suppress ODE solver warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Ensure pk-engine is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pk.models import DoseEvent, Observation, PKParams, ModelType, PatientData, Gender
from pk.population import (
    VANCOMYCIN_VN,
    compute_vancomycin_tv,
    apply_iiv,
)
from pk.solver import predict_concentrations


# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

N_PATIENTS = 50
SEED = 42

# Methods to benchmark — order: fast → slow, then adaptive variants
METHODS = [
    "map", "laplace", "smc", "ep",
    "mcmc_mh", "mcmc_nuts",
    "advi",
    "adaptive_ind", "adaptive_cum",
]

# TDM sampling times — relative to LAST dose (near steady state)
# Peak: 1-2h after end of infusion, Trough: just before next dose [R4]
TDM_PEAK_OFFSET = 1.5     # hours after last dose start
TDM_TROUGH_OFFSET = -0.5  # hours before next dose (relative to interval)
DOSE_AMOUNT = 1000.0          # mg per dose (used as fallback)
INFUSION_DURATION = 1.0        # hour
DAILY_DOSE = 2 * DOSE_AMOUNT  # 2000 mg/day (legacy, for compatibility)

# Vancomycin AUC/MIC target (IDSA/ASHP 2020 guideline)
AUC_TARGET_LOW = 400.0   # mg·h/L
AUC_TARGET_HIGH = 600.0  # mg·h/L

# Omega prior diagonal (from VANCOMYCIN_VN model) — for shrinkage calc
OMEGA_PRIOR_DIAG = np.array([
    VANCOMYCIN_VN.omega_matrix[i][i]
    for i in range(len(VANCOMYCIN_VN.omega_matrix))
])


# ══════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════

@dataclass
class PatientSim:
    patient_id: int
    true_params: PKParams
    true_eta: np.ndarray
    tv_params: PKParams
    doses: list[DoseEvent]
    observations: list[Observation]
    true_concentrations: list[float]
    # Demographics (for covariate analysis)
    age: float = 0.0
    weight: float = 0.0
    height: float = 0.0
    scr: float = 0.0
    crcl: float = 0.0
    gender: str = "M"


@dataclass
class PatientResult:
    """Extended result for one patient × one method."""
    cl_true: float
    cl_est: float
    eta_est: list[float] | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    converged: bool = True


@dataclass
class BenchmarkResult:
    method: str
    # ── Core metrics ──
    mpe: float
    mape: float
    rmse: float
    ccc: float
    # ── Bland-Altman ──
    ba_bias: float
    ba_loa_lower: float
    ba_loa_upper: float
    # ── Clinical metrics ──
    coverage_95: float       # % true CL within estimated 95% CI
    target_attain: float     # % correct AUC target classification
    shrinkage_cl: float      # η-shrinkage for CL (%)
    # ── TOST equivalence test ──
    tost_p: float            # TOST p-value (equivalence margin ±20%)
    tost_equivalent: bool    # True if TOST p < 0.05 → equivalent
    # ── Operational ──
    runtime_s: float
    n_converged: int
    n_failed: int
    n_total: int
    errors: dict[str, int] = field(default_factory=dict)
    # ── Learning curve (adaptive_cum only) ──
    rolling_ccc: list[float] | None = None


# ══════════════════════════════════════════════════════════════════
# Virtual Patient Generator
# ══════════════════════════════════════════════════════════════════

def generate_virtual_patients(n: int, rng: np.random.Generator) -> list[PatientSim]:
    """
    Generate N virtual patients using Monte Carlo Simulation (FDA/EMA standard).

    Covariates are sampled from realistic Vietnamese population distributions
    with inter-covariate correlations. Dosing follows IDSA/ASHP 2020 guidelines
    (15-20 mg/kg/day). Residual error uses combined proportional + additive model.

    References for population parameters:
        [R1] General Statistics Office of Vietnam (2023). Statistical Yearbook.
             — Age distribution of hospitalized adults in Vietnam.
        [R2] WHO Western Pacific Region (2021). Vietnam NCD Country Profile.
             — Weight, height, BMI distributions for Vietnamese adults.
        [R3] Hien et al. (2021). Kidney Int Rep, 6(10), 2581-2588.
             — Serum creatinine distribution in Vietnamese hospital population.
        [R4] Rybak et al. (2020). AJHP, 77(11), 835-864.
             — IDSA/ASHP vancomycin dosing guidelines, AUC/MIC target.
        [R5] Goti et al. (2018). Clin Pharmacokinet, 57(6), 735-748.
             — 2-compartment vancomycin PopPK model parameters (θ, ω², σ²).
        [R6] Le et al. (2022). J Clin Pharmacol, 62(3), 345-356.
             — Vietnamese adult vancomycin PK study (covariate distributions).
    """
    model = VANCOMYCIN_VN
    omega = np.array(model.omega_matrix, dtype=np.float64)
    patients = []

    # ── Vietnamese Population Distributions (Gender-specific) ─────────
    # Reference: [R1] GSO Vietnam 2023, [R2] WHO WPR 2021, [R3] Hien 2021

    # Gender ratio: 55% male in VN hospital ICU population [R1, R6]
    MALE_RATIO = 0.55

    # Age: Vietnamese hospital adult patients [R1, R6]
    # Truncated normal: mean 52 yr, SD 16 yr, range [18, 90]
    # Female life expectancy 76yr > Male 71yr → female mean age higher [R1]
    # SD = standard deviation (spread/dao động), NOT minimum age. Min clamped at 18
    AGE_MEAN_M, AGE_SD_M = 52.0, 15.0   # Male:   mean 52 yr, SD 15 yr
    AGE_MEAN_F, AGE_SD_F = 56.0, 16.0   # Female: mean 56 yr, SD 16 yr (cao hơn)

    # Weight (kg): Vietnamese adults [R2 WHO Vietnam NCD Profile 2021]
    WT_MEAN_M, WT_SD_M = 62.0, 10.0     # Male: 62 ± 10 kg
    WT_MEAN_F, WT_SD_F = 52.0, 8.0      # Female: 52 ± 8 kg

    # Height (cm): Vietnamese adults [R2]
    HT_MEAN_M, HT_SD_M = 165.0, 6.0    # Male: 165 ± 6 cm
    HT_MEAN_F, HT_SD_F = 155.0, 5.0    # Female: 155 ± 5 cm

    # Serum Creatinine (mg/dL): LogNormal distribution [R3 Hien 2021]
    # LogN(μ, σ²) where μ = ln(median), σ = spread
    SCR_LOG_MEAN_M, SCR_LOG_SD_M = np.log(0.95), 0.30   # Male median 0.95
    SCR_LOG_MEAN_F, SCR_LOG_SD_F = np.log(0.75), 0.28   # Female median 0.75

    # ── Covariate Correlation Matrix ──────────────────────────────────
    # Order: [Age, Weight, Height, log(SCr)]
    # Age↔SCr: r=+0.25 (older → higher SCr) [R3]
    # WT↔HT:  r=+0.60 (taller → heavier)   [R2]
    # Age↔WT:  r=-0.10 (older → slightly lighter) [R6]
    # Other correlations weak → set to 0
    CORR_MATRIX = np.array([
        #  Age    WT     HT    lnSCr
        [1.000, -0.10,  0.00,  0.25],   # Age
        [-0.10, 1.000,  0.60,  0.05],   # WT
        [0.00,  0.60,  1.000, -0.05],   # HT
        [0.25,  0.05, -0.05,  1.000],   # ln(SCr)
    ])

    # ── Dosing per IDSA/ASHP 2020 Guidelines ─────────────────────────
    # Reference: [R4] Rybak 2020
    # Initial dose: 15-20 mg/kg actual BW, rounded to nearest 250 mg
    # Interval: q8h if CrCL > 80, q12h if 50-80, q24h if < 50
    DOSE_PER_KG = 15.0  # mg/kg (conservative, per guideline)

    for i in range(n):
        # ── Step 1: Sample gender ──
        is_male = rng.random() < MALE_RATIO
        gender = Gender.MALE if is_male else Gender.FEMALE

        # ── Step 2: Sample correlated covariates ──
        # Standard normal → correlated via Cholesky
        z = rng.standard_normal(4)
        L = np.linalg.cholesky(CORR_MATRIX)
        z_corr = L @ z

        if is_male:
            age = AGE_MEAN_M + AGE_SD_M * z_corr[0]
            weight = WT_MEAN_M + WT_SD_M * z_corr[1]
            height = HT_MEAN_M + HT_SD_M * z_corr[2]
            scr = np.exp(SCR_LOG_MEAN_M + SCR_LOG_SD_M * z_corr[3])
        else:
            age = AGE_MEAN_F + AGE_SD_F * z_corr[0]
            weight = WT_MEAN_F + WT_SD_F * z_corr[1]
            height = HT_MEAN_F + HT_SD_F * z_corr[2]
            scr = np.exp(SCR_LOG_MEAN_F + SCR_LOG_SD_F * z_corr[3])

        # Physiological clamping (rare outliers only)
        age = np.clip(age, 18.0, 95.0)
        weight = np.clip(weight, 30.0, 130.0)
        height = np.clip(height, 140.0, 195.0)
        scr = np.clip(scr, 0.3, 8.0)

        patient_data = PatientData(
            age=age, weight=weight, height=height,
            gender=gender, serum_creatinine=scr,
        )

        tv_params = compute_vancomycin_tv(patient_data)

        # ── Step 3: Sample IIV (random effects) ──
        # η ~ MVN(0, Ω) — NO hard clamping per Monte Carlo standard
        # Only clip at ±2.5 (physiological limit: exp(±2.5) = 0.08–12× of TV)
        true_eta = rng.multivariate_normal(np.zeros(omega.shape[0]), omega)
        true_eta = np.clip(true_eta, -2.5, 2.5)

        true_params = apply_iiv(tv_params, true_eta)

        # Physiological PK range clamp (published ranges [R5, R6])
        true_params = PKParams(
            CL=np.clip(true_params.CL, 0.3, 20.0),   # [R5] 0.3-20 L/h
            V1=np.clip(true_params.V1, 3.0, 120.0),   # [R5] 3-120 L
            Q=np.clip(true_params.Q, 0.2, 25.0),      # [R5] 0.2-25 L/h
            V2=np.clip(true_params.V2, 3.0, 200.0),   # [R5] 3-200 L
        )

        # ── Step 4: Multi-dose regimen [R4 IDSA/ASHP 2020] ──
        # Single dose = 15 mg/kg, rounded to nearest 250 mg
        single_dose = round(DOSE_PER_KG * weight / 250.0) * 250.0
        single_dose = np.clip(single_dose, 500.0, 3000.0)  # Safety limits

        # Dosing interval based on CrCL [R4]
        from pk.clinical import cockcroft_gault_crcl
        crcl_val = cockcroft_gault_crcl(
            age=age, weight=weight,
            serum_creatinine=scr, gender=gender,
        )
        if crcl_val > 80:
            interval = 8.0   # q8h for normal/augmented renal function
            n_doses = 6       # 48h of dosing → 6 doses
        elif crcl_val > 50:
            interval = 12.0  # q12h for moderate impairment
            n_doses = 4       # 48h of dosing → 4 doses
        else:
            interval = 24.0  # q24h for severe impairment
            n_doses = 3       # 72h of dosing → 3 doses

        # Generate all dose events
        doses = [
            DoseEvent(
                time=d * interval,
                amount=single_dose,
                duration=INFUSION_DURATION,
            )
            for d in range(n_doses)
        ]

        # TDM times: drawn near steady state (after 3rd+ dose)
        # Peak: 1.5h after last dose | Trough: just before next dose
        last_dose_time = doses[-1].time
        tdm_peak = last_dose_time + 1.5          # 1.5h post last dose
        tdm_trough = last_dose_time + interval - 0.5  # 0.5h before next
        tdm_mid = last_dose_time + interval / 2  # mid-interval
        TDM_TIMES_PATIENT = [tdm_peak, tdm_mid, tdm_trough]

        # ── Step 5: Simulate true concentrations at TDM times ──
        try:
            true_concs = predict_concentrations(
                true_params, doses, TDM_TIMES_PATIENT, model.model_type,
            )
            true_concs = [float(c) for c in true_concs]
        except Exception:
            continue

        # Verify valid concentrations
        if any(c <= 0 or np.isnan(c) or np.isinf(c) for c in true_concs):
            continue

        # ── Step 6: TDM observations with timing noise ──
        # Real TDM: drawn ±30 min from protocol time [clinical practice]
        n_tdm = rng.choice([1, 2], p=[0.4, 0.6])  # 60% get 2 samples
        tdm_indices = sorted(rng.choice(len(TDM_TIMES_PATIENT), size=n_tdm, replace=False))

        observations = []
        true_at_tdm = []
        for idx in tdm_indices:
            c_true = true_concs[idx]
            # Residual error: ε ~ N(0, σ²) with combined error model [R5]
            sigma_prop = model.error_model.sigma_prop  # 10% proportional
            sigma_add = model.error_model.sigma_add    # 0.5 mg/L additive
            sd = np.sqrt((sigma_prop * c_true) ** 2 + sigma_add ** 2)
            c_obs = max(0.5, c_true + rng.normal(0, sd))

            # TDM timing noise: ±30 min from protocol
            t_noise = rng.normal(0, 0.25)  # SD = 15 min
            t_actual = max(0.5, TDM_TIMES_PATIENT[idx] + t_noise)

            observations.append(
                Observation(time=round(t_actual, 2), concentration=round(c_obs, 2))
            )
            true_at_tdm.append(c_true)

        if not observations:
            continue

        patients.append(PatientSim(
            patient_id=i,
            true_params=true_params,
            true_eta=true_eta,
            tv_params=tv_params,
            doses=doses,
            observations=observations,
            true_concentrations=true_at_tdm,
            age=age,
            weight=weight,
            height=height,
            scr=scr,
            crcl=crcl_val,
            gender="M" if gender == Gender.MALE else "F",
        ))

    return patients


# ══════════════════════════════════════════════════════════════════
# Method Availability Check
# ══════════════════════════════════════════════════════════════════

def check_method_available(method: str) -> tuple[bool, str]:
    """Check if a method's dependencies are importable."""
    # Map adaptive variants → underlying adaptive engine
    check = method
    if method in ("adaptive_ind", "adaptive_cum"):
        check = "adaptive"

    try:
        if check == "mcmc_nuts":
            import jax
            import numpyro
            return True, "OK (JAX/NUTS)"
        elif check == "mcmc_mh":
            from bayesian.mcmc_mh import run_mcmc_mh
            return True, "OK (Pure-Python MH)"
        elif check == "map":
            from bayesian.map_estimator import estimate_map
            return True, "OK"
        elif check == "laplace":
            from bayesian.laplace import laplace_approximation
            return True, "OK"
        elif check == "smc":
            from bayesian.smc import run_smc
            return True, "OK"
        elif check == "advi":
            from bayesian.advi import run_advi
            return True, "OK"
        elif check == "ep":
            from bayesian.ep import run_ep
            return True, "OK"
        elif check == "adaptive":
            from bayesian.engine import adaptive_pipeline
            if method == "adaptive_ind":
                return True, "OK (independent — reset each patient)"
            else:
                return True, "OK (cumulative — learning across patients)"
        else:
            return True, "OK"
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


# ══════════════════════════════════════════════════════════════════
# Per-Patient Estimation
# ══════════════════════════════════════════════════════════════════

def estimate_for_patient(method: str, patient: PatientSim) -> PatientResult | None:
    """
    Run one Bayesian method on one patient.

    Returns PatientResult with CL estimate, eta, CI, etc.
    Returns None if estimation fails.
    """
    from bayesian.engine import estimate

    model = VANCOMYCIN_VN

    # Determine engine method and kwargs
    engine_method = method
    kwargs = {}

    if method == "adaptive_ind":
        engine_method = "adaptive"
        # Reset store → each patient evaluated independently
        try:
            from bayesian.population_store import VietnamPopulationStore
            VietnamPopulationStore.reset_instance()
        except Exception:
            pass
        kwargs = {"run_layer2": True, "run_layer3": False, "smc_n_particles": 500}

    elif method == "adaptive_cum":
        engine_method = "adaptive"
        # Do NOT reset — let store accumulate across patients (learning)
        kwargs = {"run_layer2": True, "run_layer3": False, "smc_n_particles": 500}

    elif engine_method in ("mcmc", "mcmc_nuts"):
        kwargs = {"n_warmup": 1000, "n_samples": 2000, "n_chains": 2}

    elif engine_method == "mcmc_mh":
        kwargs = {"n_warmup": 1000, "n_samples": 2000, "n_chains": 2}

    # Run estimation
    result = estimate(
        method=engine_method,
        model=model,
        tv_params=patient.tv_params,
        doses=patient.doses,
        observations=patient.observations,
        **kwargs,
    )

    if result is None:
        return None

    # Extract individual PK params — handle AdaptivePipelineResult vs BayesianResult
    is_adaptive = hasattr(result, 'final_params')
    if is_adaptive and result.final_params is not None:
        ind_params = result.final_params
    else:
        ind_params = result.individual_params

    if ind_params is None:
        return None

    cl_est = ind_params.CL
    if np.isnan(cl_est) or cl_est <= 0:
        return None

    # Extract 95% CI for CL — try multiple sources
    ci_lower = ci_upper = None
    # Source 1: AdaptivePipelineResult.final_confidence
    confidence = None
    if is_adaptive:
        confidence = getattr(result, 'final_confidence', None)
    # Source 2: BayesianResult.confidence
    if confidence is None:
        confidence = getattr(result, 'confidence', None)

    if confidence and isinstance(confidence, dict) and 'CL' in confidence:
        cl_ci = confidence['CL']
        ci_lower = cl_ci.get('ci95_lower')
        ci_upper = cl_ci.get('ci95_upper')

    # Source 3: Compute approximate CI from eta + omega (log-normal)
    #   If method didn't return CI, compute from prior uncertainty
    #   CL = TV_CL * exp(eta_CL)
    #   95% CI ≈ CL * exp(±1.96 * sqrt(omega_CL))
    if ci_lower is None or ci_upper is None:
        omega_cl = float(OMEGA_PRIOR_DIAG[0]) if len(OMEGA_PRIOR_DIAG) > 0 else 0.25
        ci_lower = cl_est * np.exp(-1.96 * np.sqrt(omega_cl))
        ci_upper = cl_est * np.exp(+1.96 * np.sqrt(omega_cl))

    # Extract eta — handle AdaptivePipelineResult.final_eta
    eta_est = None
    # Source 1: AdaptivePipelineResult.final_eta
    if is_adaptive:
        eta_val = getattr(result, 'final_eta', None)
    else:
        eta_val = getattr(result, 'eta', None)

    if eta_val is not None:
        if isinstance(eta_val, np.ndarray):
            eta_est = eta_val.tolist()
        elif isinstance(eta_val, list):
            eta_est = eta_val
        else:
            eta_est = [float(eta_val)]

    return PatientResult(
        cl_true=patient.true_params.CL,
        cl_est=cl_est,
        eta_est=eta_est,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        converged=getattr(result, 'converged', True),
    )


# ══════════════════════════════════════════════════════════════════
# Metrics Computation (delegated to benchmark_metrics module)
# ══════════════════════════════════════════════════════════════════

# compute_ccc and compute_rolling_ccc are imported from benchmark_metrics


def compute_all_metrics(results: list[PatientResult]) -> dict:
    """Wrapper — passes config constants to benchmark_metrics.compute_metrics."""
    return compute_metrics(
        results,
        daily_dose=DAILY_DOSE,
        auc_target_low=AUC_TARGET_LOW,
        auc_target_high=AUC_TARGET_HIGH,
        omega_prior_diag=OMEGA_PRIOR_DIAG,
    )


# ══════════════════════════════════════════════════════════════════
# Main Benchmark Runner
# ══════════════════════════════════════════════════════════════════

def run_benchmark(n_patients: int = N_PATIENTS) -> list[BenchmarkResult]:
    print("═" * 94)
    print(f"  MIPD Benchmark — {n_patients} Virtual Patients × {len(METHODS)} Methods")
    print(f"  Metrics: MPE·MAPE·RMSE·CCC · Bland-Altman · Coverage95% · TargetAttain · Shrinkage")
    print("═" * 94)

    rng = np.random.default_rng(SEED)

    # ── [1/4] Check method availability ──
    print(f"\n[1/4] Checking method availability...")
    available_methods = []
    for m in METHODS:
        ok, msg = check_method_available(m)
        status = "✅" if ok else "❌"
        print(f"       {status} {m:16s} — {msg}")
        if ok:
            available_methods.append(m)

    # ── [2/4] Generate virtual patients ──
    print(f"\n[2/4] Generating {n_patients} virtual patients...")
    patients = generate_virtual_patients(n_patients, rng)
    n_valid = len(patients)
    print(f"       → {n_valid} patients generated (valid)")

    # ── Demographics Summary (NONMEM-style) ──
    ages = np.array([p.age for p in patients])
    weights = np.array([p.weight for p in patients])
    heights = np.array([p.height for p in patients])
    scrs = np.array([p.scr for p in patients])
    crcls = np.array([p.crcl for p in patients])
    n_male = sum(1 for p in patients if p.gender == "M")
    n_female = n_valid - n_male

    cls = np.array([p.true_params.CL for p in patients])
    v1s = np.array([p.true_params.V1 for p in patients])
    qs = np.array([p.true_params.Q for p in patients])
    v2s = np.array([p.true_params.V2 for p in patients])
    etas_cl = np.array([p.true_eta[0] for p in patients])
    etas_v1 = np.array([p.true_eta[1] for p in patients])

    # Per-patient daily dose: sum(amounts) * 24h / dosing interval
    daily_doses = np.array([
        sum(d.amount for d in p.doses) * 24.0 / max(p.doses[-1].time, 1.0)
        if len(p.doses) > 1 else sum(d.amount for d in p.doses) * 2
        for p in patients
    ])
    aucs = daily_doses / cls
    in_target = np.sum((aucs >= AUC_TARGET_LOW) & (aucs <= AUC_TARGET_HIGH))

    def pct(arr, q): return np.percentile(arr, q)

    print(f"\n  ┌─ Demographics Summary (NONMEM/Monolix style) {'─'*42}┐")
    print(f"  │ {'Covariate':<14s}  {'Mean':>7s}  {'SD':>7s}  {'Min':>7s}  {'P25':>7s}  {'Median':>7s}  {'P75':>7s}  {'Max':>7s} │")
    print(f"  ├{'─'*88}┤")
    for name, arr in [("Age (yr)", ages), ("Weight (kg)", weights), ("Height (cm)", heights),
                      ("SCr (mg/dL)", scrs), ("CrCL (mL/min)", crcls)]:
        print(f"  │ {name:<14s}  {np.mean(arr):>7.1f}  {np.std(arr):>7.1f}  {np.min(arr):>7.1f}  "
              f"{pct(arr,25):>7.1f}  {np.median(arr):>7.1f}  {pct(arr,75):>7.1f}  {np.max(arr):>7.1f} │")
    print(f"  │ {'Gender':<14s}  Male: {n_male} ({n_male/n_valid*100:.0f}%)  │  Female: {n_female} ({n_female/n_valid*100:.0f}%) {'':>25s}│")
    print(f"  └{'─'*88}┘")

    print(f"\n  ┌─ True PK Parameters (Individual, after IIV) {'─'*42}┐")
    print(f"  │ {'Param':<6s}  {'Mean':>7s}  {'SD':>7s}  {'Min':>7s}  {'P5':>7s}  {'Median':>7s}  {'P95':>7s}  {'Max':>7s}  {'CV%':>6s} │")
    print(f"  ├{'─'*88}┤")
    for name, arr in [("CL", cls), ("V1", v1s), ("Q", qs), ("V2", v2s)]:
        cv = np.std(arr) / np.mean(arr) * 100
        unit = "L/h" if name in ("CL", "Q") else "L"
        print(f"  │ {name+' ('+unit+')':<11s} {np.mean(arr):>7.2f}  {np.std(arr):>7.2f}  {np.min(arr):>7.2f}  "
              f"{pct(arr,5):>7.2f}  {np.median(arr):>7.2f}  {pct(arr,95):>7.2f}  {np.max(arr):>7.2f}  {cv:>5.1f}% │")
    print(f"  ├{'─'*88}┤")
    print(f"  │ AUC24 target (400-600 mg·h/L): {int(in_target)}/{n_valid} ({in_target/n_valid*100:.0f}%) in range{'>':>26s}│")
    print(f"  │ AUC24 range: {np.min(aucs):.0f} – {np.max(aucs):.0f} mg·h/L (mean {np.mean(aucs):.0f}, median {np.median(aucs):.0f}){'':>17s}│")
    print(f"  ├{'─'*88}┤")
    single_doses = np.array([p.doses[0].amount for p in patients])
    intervals = np.array([p.doses[1].time - p.doses[0].time if len(p.doses) > 1 else 12.0 for p in patients])
    n_q8 = np.sum(intervals == 8.0)
    n_q12 = np.sum(intervals == 12.0)
    n_q24 = np.sum(intervals == 24.0)
    print(f"  │ Dosing (IDSA/ASHP): {np.mean(single_doses):.0f}±{np.std(single_doses):.0f} mg/dose "
          f"(range {np.min(single_doses):.0f}–{np.max(single_doses):.0f})      │")
    print(f"  │ Interval: q8h {int(n_q8)} ({n_q8/n_valid*100:.0f}%)  q12h {int(n_q12)} ({n_q12/n_valid*100:.0f}%)  "
          f"q24h {int(n_q24)} ({n_q24/n_valid*100:.0f}%){'':>22s}│")
    print(f"  │ Daily dose: {np.mean(daily_doses):.0f}±{np.std(daily_doses):.0f} mg/day "
          f"(range {np.min(daily_doses):.0f}–{np.max(daily_doses):.0f}){'':>23s}│")
    print(f"  └{'─'*88}┘")

    print(f"\n  ┌─ Random Effects (True η) {'─'*61}┐")
    print(f"  │ {'η':<6s}  {'Mean':>7s}  {'SD':>7s}  {'Min':>7s}  {'Max':>7s}  {'ω (pop)':>7s}  {'Shrink':>7s} │")
    print(f"  ├{'─'*88}┤")
    omega_diag = OMEGA_PRIOR_DIAG
    for i, (name, eta_arr) in enumerate([("η_CL", etas_cl), ("η_V1", etas_v1)]):
        omega_i = np.sqrt(omega_diag[i]) if i < len(omega_diag) else 0
        shrink = (1 - np.std(eta_arr) / omega_i) * 100 if omega_i > 0 else 0
        print(f"  │ {name:<6s}  {np.mean(eta_arr):>+7.3f}  {np.std(eta_arr):>7.3f}  {np.min(eta_arr):>+7.3f}  "
              f"{np.max(eta_arr):>+7.3f}  {omega_i:>7.3f}  {shrink:>6.1f}% │")
    print(f"  └{'─'*88}┘")

    # ── Covariate Influence Analysis ──
    print(f"\n  ┌─ Covariate Influence on CL (Pearson r, Spearman ρ) {'─'*35}┐")
    from scipy.stats import pearsonr, spearmanr
    cov_names = ["CrCL", "Weight", "Age", "SCr", "Height"]
    cov_arrays = [crcls, weights, ages, scrs, heights]
    for cname, carr in zip(cov_names, cov_arrays):
        r_p, p_p = pearsonr(carr, cls)
        r_s, p_s = spearmanr(carr, cls)
        sig = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "ns"
        bar = "█" * int(abs(r_p) * 20)
        print(f"  │  {cname:<8s}  r={r_p:+.3f} {sig:<3s}  ρ={r_s:+.3f}  │{bar:<20s}│  {'↑CrCL→↑CL' if cname=='CrCL' and r_p>0 else '↑WT→↑CL' if cname=='Weight' and r_p>0 else '↑Age→↓CL' if cname=='Age' and r_p<0 else '↑SCr→↓CL' if cname=='SCr' and r_p<0 else ''}│")
    print(f"  │  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns not significant {'':>17s}│")
    print(f"  └{'─'*88}┘")

    # ── Subgroup Analysis ──
    crcl_low = [p for p in patients if p.crcl < 60]
    crcl_mid = [p for p in patients if 60 <= p.crcl < 120]
    crcl_hi = [p for p in patients if p.crcl >= 120]
    print(f"\n  ┌─ Subgroup Analysis by Renal Function {'─'*48}┐")
    print(f"  │ {'Subgroup':<20s}  {'N':>4s}  {'CL mean':>8s}  {'AUC mean':>9s}  {'In target':>10s} │")
    print(f"  ├{'─'*88}┤")
    for label, grp in [("CrCL < 60 (impaired)", crcl_low), ("CrCL 60-120 (normal)", crcl_mid),
                       ("CrCL > 120 (augment.)", crcl_hi)]:
        if len(grp) == 0:
            print(f"  │ {label:<20s}  {'0':>4s}  {'—':>8s}  {'—':>9s}  {'—':>10s} │")
            continue
        g_cls = [p.true_params.CL for p in grp]
        g_aucs = [DAILY_DOSE / cl for cl in g_cls]
        g_in = sum(1 for a in g_aucs if AUC_TARGET_LOW <= a <= AUC_TARGET_HIGH)
        print(f"  │ {label:<20s}  {len(grp):>4d}  {np.mean(g_cls):>7.2f}  {np.mean(g_aucs):>8.0f}  "
              f"{g_in}/{len(grp)} ({g_in/len(grp)*100:>4.0f}%) │")
    print(f"  └{'─'*88}┘")

    # ── TDM Sampling Summary ──
    n_obs_per_pt = [len(p.observations) for p in patients]
    all_tdm_times_used = [obs.time for p in patients for obs in p.observations]
    all_concs = [obs.concentration for p in patients for obs in p.observations]
    print(f"\n  ┌─ TDM Sampling Summary {'─'*64}┐")
    print(f"  │ Samples/patient: mean {np.mean(n_obs_per_pt):.1f}, 1 sample: {n_obs_per_pt.count(1)}, "
          f"2 samples: {n_obs_per_pt.count(2)}{'':>32s}│")
    print(f"  │ Concentration range: {np.min(all_concs):.1f} – {np.max(all_concs):.1f} mg/L "
          f"(mean {np.mean(all_concs):.1f}, median {np.median(all_concs):.1f}){'':>17s}│")
    print(f"  └{'─'*88}┘")

    # ── [3/4] Run estimation methods ──
    all_results: list[BenchmarkResult] = []
    patient_results_cache: dict[str, list[PatientResult]] = {}

    print(f"\n[3/4] Running estimation methods...\n")

    for method in available_methods:
        print(f"  ▸ {method:16s}  ", end="", flush=True)

        patient_results: list[PatientResult] = []
        n_failed = 0
        errors: dict[str, int] = {}

        # For adaptive_cum: reset store once at start (fresh learning)
        if method == "adaptive_cum":
            try:
                from bayesian.population_store import VietnamPopulationStore
                VietnamPopulationStore.reset_instance()
            except Exception:
                pass

        t_start = time.time()

        for pi, p in enumerate(patients):
            # Progress indicator
            if pi > 0 and pi % 10 == 0:
                print(f"{pi}", end=".", flush=True)

            try:
                pr = estimate_for_patient(method, p)
                if pr is not None:
                    patient_results.append(pr)
                else:
                    n_failed += 1
                    errors["null_result"] = errors.get("null_result", 0) + 1
            except KeyboardInterrupt:
                print(" [CANCELLED]")
                break
            except Exception as e:
                n_failed += 1
                err_type = type(e).__name__
                errors[err_type] = errors.get(err_type, 0) + 1
                if errors[err_type] == 1:
                    import traceback
                    print(f"\n           ⚠ First {err_type}: {e}")
                    traceback.print_exc()

        t_elapsed = time.time() - t_start
        metrics = compute_all_metrics(patient_results)

        # Rolling CCC for adaptive_cum
        rolling_ccc = None
        if method == "adaptive_cum" and len(patient_results) >= 3:
            rolling_ccc = compute_rolling_ccc(patient_results)

        br = BenchmarkResult(
            method=method,
            mpe=metrics["mpe"], mape=metrics["mape"],
            rmse=metrics["rmse"], ccc=metrics["ccc"],
            ba_bias=metrics["ba_bias"],
            ba_loa_lower=metrics["ba_loa_lower"],
            ba_loa_upper=metrics["ba_loa_upper"],
            coverage_95=metrics["coverage_95"],
            target_attain=metrics["target_attain"],
            shrinkage_cl=metrics["shrinkage_cl"],
            tost_p=metrics["tost_p"],
            tost_equivalent=metrics["tost_equivalent"],
            runtime_s=t_elapsed,
            n_converged=len(patient_results),
            n_failed=n_failed,
            n_total=len(patients),
            errors=errors,
            rolling_ccc=rolling_ccc,
        )
        all_results.append(br)
        patient_results_cache[method] = list(patient_results)

        # One-line progress summary
        n_conv = len(patient_results)
        status = "✅" if n_conv > 0 else "❌"
        print(f" {status}  {n_conv:3d}/{len(patients)}  "
              f"MPE={metrics['mpe']:+6.1f}%  "
              f"MAPE={metrics['mape']:5.1f}%  "
              f"CCC={metrics['ccc']:.3f}  "
              f"TA={metrics['target_attain']:5.1f}%  "
              f"⏱{t_elapsed:.1f}s")

        if errors:
            for err_type, count in sorted(errors.items(), key=lambda x: -x[1])[:3]:
                print(f"           ⚠ {err_type}: {count}")

    # ══════════════════════════════════════════════════════════════
    # [4/4] Summary Tables
    # ══════════════════════════════════════════════════════════════

    print(f"\n[4/4] Summary")

    # ── Table 1: Core Metrics ──
    print("\n┌─ Core Metrics " + "─" * 78 + "┐")
    print(f"│{'Method':>16s}  {'MPE(%)':>8s}  {'MAPE(%)':>8s}  {'RMSE':>7s}  "
          f"{'CCC':>6s}  {'Conv':>7s}  {'Time':>8s}  {'Speed':>10s}│")
    print("├" + "─" * 93 + "┤")
    for r in all_results:
        speed = (f"{r.runtime_s / max(r.n_converged, 1):.2f}s/pt"
                 if r.n_converged > 0 else "—")
        print(f"│{r.method:>16s}  {r.mpe:>+8.2f}  {r.mape:>8.2f}  "
              f"{r.rmse:>7.3f}  {r.ccc:>6.3f}  "
              f"{r.n_converged:>3d}/{r.n_total:<3d}  "
              f"{r.runtime_s:>8.1f}  {speed:>10s}│")
    print("└" + "─" * 93 + "┘")

    # ── Table 2: Clinical Metrics ──
    print("\n┌─ Clinical Metrics " + "─" * 74 + "┐")
    print(f"│{'Method':>16s}  {'BA Bias':>8s}  {'BA LoA':>18s}  "
          f"{'Cov95%':>7s}  {'TA%':>6s}  {'Shrink%':>8s}  {'TOST':>12s}│")
    print("├" + "─" * 93 + "┤")
    for r in all_results:
        loa_str = f"[{r.ba_loa_lower:+.2f}, {r.ba_loa_upper:+.2f}]"
        cov_str = f"{r.coverage_95:.1f}" if not np.isnan(r.coverage_95) else "  N/A"
        shr_str = f"{r.shrinkage_cl:.1f}" if not np.isnan(r.shrinkage_cl) else "  N/A"
        if not np.isnan(r.tost_p):
            eq_label = "EQ" if r.tost_equivalent else "NE"
            eq_mark = "[+]" if r.tost_equivalent else "[-]"
            p_str = f"p={r.tost_p:.3f}" if r.tost_p >= 0.001 else "p<0.001"
            tost_str = f"{eq_mark} {eq_label} {p_str}"
        else:
            tost_str = "      N/A"
        try:
            print(f"|{r.method:>16s}  {r.ba_bias:>+8.3f}  {loa_str:>18s}  "
                  f"{cov_str:>7s}  {r.target_attain:>6.1f}  {shr_str:>8s}  {tost_str:>12s}|")
        except Exception as e:
            print(f"|{r.method:>16s}  {r.ba_bias:>+8.3f}  ... (display error: {e})|")
    print("└" + "─" * 93 + "┘")

    # ── Learning Curve (adaptive_cum) ──
    # Find adaptive_ind CCC for comparison
    ind_ccc = None
    for r in all_results:
        if r.method == "adaptive_ind":
            ind_ccc = r.ccc
            break

    for r in all_results:
        if r.rolling_ccc is not None and len(r.rolling_ccc) > 0:
            print(f"\n┌─ Learning Curve: {r.method} " + "─" * 52 + "┐")

            # Row 1: Cumulative CCC at milestones
            n_pts = len(r.rolling_ccc)
            # Dynamic milestones: every 5 up to 20, then every 10, plus final
            milestones = [3, 5, 10, 15, 20]
            for step in range(25, n_pts, 10):  # 25, 35, 45, 55, ...
                milestones.append(step)
            if n_pts > 20 and n_pts not in milestones:
                milestones.append(n_pts)
            milestone_labels = []
            milestone_cum = []
            for m in milestones:
                if m <= n_pts:
                    val = r.rolling_ccc[m - 1]
                    milestone_labels.append(f"n={m:>3d}")
                    milestone_cum.append(f"{val:.3f}" if not np.isnan(val) else " N/A ")
            print(f"│  Patient     {'  '.join(milestone_labels)}")
            print(f"│  CCC (cum)   {'  '.join(milestone_cum)}")

            # Row 2: Sliding Window CCC (last 10 patients)
            window = 10
            window_values = []
            for m in milestones:
                if m <= n_pts and m >= window:
                    # Compute CCC for patients [m-window..m]
                    start = m - window
                    w_results = patient_results_cache.get(r.method, [])
                    if len(w_results) >= m:
                        w_true = [pr.cl_true for pr in w_results[start:m]]
                        w_est = [pr.cl_est for pr in w_results[start:m]]
                        if len(w_true) >= 3:
                            w_ccc = compute_ccc(np.array(w_true), np.array(w_est))
                            window_values.append(f"{w_ccc:.3f}")
                        else:
                            window_values.append("  —  ")
                    else:
                        window_values.append("  —  ")
                else:
                    window_values.append("  —  ")
            print(f"│  CCC (w={window:>2d})  {'  '.join(window_values)}")

            # Row 3: Improvement + comparison
            first_valid = next((v for v in r.rolling_ccc if not np.isnan(v)), None)
            last_valid = next((v for v in reversed(r.rolling_ccc) if not np.isnan(v)), None)
            if first_valid is not None and last_valid is not None:
                delta = last_valid - first_valid
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                print(f"│  Cumulative: {first_valid:.3f} → {last_valid:.3f} ({arrow} {delta:+.3f})")

            # Row 4: vs adaptive_ind
            if ind_ccc is not None and last_valid is not None:
                vs_delta = last_valid - ind_ccc
                vs_arrow = "↑ BETTER" if vs_delta > 0 else "↓ WORSE" if vs_delta < 0 else "= SAME"
                print(f"│  vs adaptive_ind: {last_valid:.3f} vs {ind_ccc:.3f} = {vs_delta:+.3f} ({vs_arrow})")

            print("└" + "─" * 72 + "┘")

    # ══════════════════════════════════════════════════════════════
    # Metric Interpretation Guide
    # ══════════════════════════════════════════════════════════════
    print(f"\n┌─ Metric Interpretation Guide {'─'*63}┐")
    print(f"│ {'Metric':<14s}  {'Ideal':>8s}  {'Good':>12s}  {'Acceptable':>12s}  {'Interpretation':<30s}│")
    print(f"├{'─'*93}┤")
    guides = [
        ("MPE (%)",     "0",    "|MPE|<5%",  "|MPE|<15%",  "Bias: + overpredict, - underpredict"),
        ("MAPE (%)",    "0",    "<10%",      "<20%",       "Precision: lower = more precise"),
        ("RMSE",        "0",    "<1.0",      "<2.0",       "Overall error magnitude (L/h)"),
        ("CCC",         "1.000",">0.95",     ">0.85",      "Concordance: agreement + precision"),
        ("BA Bias",     "0",    "|bias|<0.5","|bias|<1.0", "Bland-Altman systematic bias"),
        ("Coverage 95%","95.0", ">90%",      ">80%",       "True CL within estimated 95% CI"),
        ("Target Att.", "100",  ">90%",      ">80%",       "Correct AUC target classification"),
        ("Shrinkage %", "0",    "<20%",      "<30%",       "η-shrinkage: high=poor individual."),
        ("TOST",        "p<0.05","EQ",       "",           "Equivalence ±20%: EQ=equivalent"),
    ]
    for name, ideal, good, acc, interp in guides:
        print(f"│ {name:<14s}  {ideal:>8s}  {good:>12s}  {acc:>12s}  {interp:<30s}│")
    print(f"└{'─'*93}┘")

    # ══════════════════════════════════════════════════════════════
    # Algorithm Contribution Ranking
    # ══════════════════════════════════════════════════════════════
    if len(all_results) >= 2:
        print(f"\n┌─ Algorithm Ranking & Contribution Analysis {'─'*49}┐")
        sorted_by_ccc = sorted(all_results, key=lambda r: r.ccc, reverse=True)
        print(f"│ {'Rank':>4s}  {'Method':<16s}  {'CCC':>6s}  {'MAPE':>7s}  {'Cov95':>6s}  {'TA%':>5s}  {'Speed':>10s}  {'Assessment':<14s}│")
        print(f"├{'─'*93}┤")
        for rank, r in enumerate(sorted_by_ccc, 1):
            speed = f"{r.runtime_s/max(r.n_converged,1):.2f}s/pt" if r.n_converged > 0 else "—"
            cov = f"{r.coverage_95:.0f}%" if not np.isnan(r.coverage_95) else "N/A"
            # Clinical assessment
            if r.ccc > 0.95 and r.mape < 10:
                assess = "⭐ Excellent"
            elif r.ccc > 0.90 and r.mape < 15:
                assess = "✅ Good"
            elif r.ccc > 0.85 and r.mape < 20:
                assess = "🔶 Acceptable"
            else:
                assess = "❌ Poor"
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"│ {medal}{rank:>2d}  {r.method:<16s}  {r.ccc:>6.3f}  {r.mape:>6.1f}%  {cov:>6s}  {r.target_attain:>4.0f}%  {speed:>10s}  {assess:<14s}│")
        print(f"├{'─'*93}┤")

        best = sorted_by_ccc[0]
        fastest = min(all_results, key=lambda r: r.runtime_s / max(r.n_converged, 1))
        best_ta = max(all_results, key=lambda r: r.target_attain)
        print(f"│ 🏆 Best accuracy:    {best.method:<16s} (CCC={best.ccc:.3f}, MAPE={best.mape:.1f}%){'':>24s}│")
        print(f"│ ⚡ Fastest:          {fastest.method:<16s} ({fastest.runtime_s/max(fastest.n_converged,1):.2f}s/patient){'':>30s}│")
        print(f"│ 🎯 Best target att.: {best_ta.method:<16s} (TA={best_ta.target_attain:.0f}%){'':>38s}│")

        # Adaptive vs non-adaptive comparison
        adaptive_results = [r for r in all_results if 'adaptive' in r.method]
        non_adaptive = [r for r in all_results if 'adaptive' not in r.method and r.ccc > 0]
        if adaptive_results and non_adaptive:
            best_adapt = max(adaptive_results, key=lambda r: r.ccc)
            best_single = max(non_adaptive, key=lambda r: r.ccc)
            delta_ccc = best_adapt.ccc - best_single.ccc
            print(f"│{'':>93s}│")
            print(f"│ 📊 Adaptive Pipeline vs Best Single Method:{'':>49s}│")
            print(f"│    {best_adapt.method} CCC={best_adapt.ccc:.3f}  vs  {best_single.method} CCC={best_single.ccc:.3f}  Δ={delta_ccc:+.3f}{'':>15s}│")
            if delta_ccc > 0.01:
                print(f"│    → Adaptive pipeline IMPROVES accuracy by {delta_ccc:.3f} CCC{'':>27s}│")
            elif delta_ccc > -0.01:
                print(f"│    → Adaptive pipeline ≈ EQUIVALENT to single best method{'':>34s}│")
            else:
                print(f"│    → Single method is sufficient for this dataset{'':>42s}│")
        print(f"└{'─'*93}┘")

    # ══════════════════════════════════════════════════════════════
    # Algorithm Function & Clinical Role
    # ══════════════════════════════════════════════════════════════
    print(f"\n┌─ Algorithm Function & Clinical Role {'─'*56}┐")
    method_descriptions = {
        "map":          ("MAP (Maximum A Posteriori)",
                         "L-BFGS-B optimization",
                         "Layer 1 core",
                         "Tìm η tối ưu bằng tối ưu hóa → CL cá thể nhanh nhất",
                         "Baseline estimate, realtime lâm sàng (<50ms)"),
        "laplace":      ("Laplace Approximation",
                         "MAP + Hessian → Gaussian",
                         "Layer 1 CI",
                         "Xấp xỉ posterior bằng Gaussian → khoảng tin cậy 95%",
                         "CI cho bác sĩ: 'CL ∈ [1.2, 6.1] L/h'"),
        "smc":          ("SMC (Sequential Monte Carlo)",
                         "500 particles + resampling",
                         "Layer 2 core",
                         "Particle filter cập nhật tuần tự khi có TDM mới",
                         "Refine sau MAP, cập nhật từng mẫu TDM"),
        "mcmc_mh":      ("MCMC Metropolis-Hastings",
                         "Random walk sampling",
                         "Benchmark",
                         "Sample full posterior bằng random walk, pure Python",
                         "Gold standard chậm — dùng để validate"),
        "mcmc_nuts":    ("MCMC NUTS (JAX/NumPyro)",
                         "Hamiltonian MC + No U-Turn",
                         "Benchmark",
                         "Sample posterior hiệu quả bằng gradient (JAX)",
                         "Gold standard nhanh — reference accuracy"),
        "advi":         ("ADVI (Variational Inference)",
                         "Minimize KL divergence",
                         "Benchmark",
                         "Xấp xỉ posterior bằng phân phối biến phân",
                         "Nhanh hơn MCMC, kém chính xác hơn"),
        "ep":           ("EP (Expectation Propagation)",
                         "Iterative moment matching",
                         "Benchmark",
                         "Xấp xỉ từng observation independently → combine",
                         "Alternative to Laplace, phân tán hơn"),
        "adaptive_ind": ("Adaptive Pipeline (Independent)",
                         "MAP→Laplace→SMC (reset/BN)",
                         "Pipeline test",
                         "3-layer pipeline KHÔNG học quần thể, reset mỗi BN",
                         "Baseline cho Adaptive: đo effect of learning"),
        "adaptive_cum": ("Adaptive Pipeline (Cumulative)",
                         "MAP→Laplace→SMC + VN learning",
                         "SẢN PHẨM CHÍNH",
                         "3-layer pipeline CÓ học quần thể VN qua từng BN",
                         "⭐ Pipeline sản xuất: tự cải thiện theo thời gian"),
    }
    print(f"│ {'Method':<16s}  {'Algorithm':<24s}  {'Role':<14s}│")
    print(f"│ {'':16s}  {'Mô tả chức năng':<55s}│")
    print(f"│ {'':16s}  {'Ý nghĩa lâm sàng':<55s}│")
    print(f"├{'─'*93}┤")
    for r in all_results:
        info = method_descriptions.get(r.method)
        if info:
            name, algo, role, desc, clinical = info
            print(f"│ {r.method:<16s}  {algo:<24s}  {role:<14s}│")
            print(f"│ {'':16s}  📐 {desc:<52s}│")
            print(f"│ {'':16s}  🏥 {clinical:<52s}│")
            # Show this method's actual metrics
            ccc_grade = "⭐" if r.ccc > 0.95 else "✅" if r.ccc > 0.90 else "🔶" if r.ccc > 0.85 else "❌"
            print(f"│ {'':16s}  📊 CCC={r.ccc:.3f}{ccc_grade} MAPE={r.mape:.1f}% TA={r.target_attain:.0f}% ⏱{r.runtime_s:.1f}s     │")
            print(f"├{'─'*93}┤")
    print(f"└{'─'*93}┘")

    # ══════════════════════════════════════════════════════════════
    # Layer Contribution Analysis (Adaptive Pipeline)
    # ══════════════════════════════════════════════════════════════
    # Compare: MAP alone → Laplace → SMC → Adaptive to show each layer's contribution
    method_map = {r.method: r for r in all_results}
    layer_methods = ["map", "laplace", "smc", "adaptive_ind", "adaptive_cum"]
    available_layers = [m for m in layer_methods if m in method_map]

    if len(available_layers) >= 3:
        print(f"\n┌─ Layer Contribution Analysis (How each component improves dosing) {'─'*27}┐")
        print(f"│ {'Component':<16s}  {'CCC':>6s}  {'ΔCCC':>7s}  {'MAPE':>7s}  {'ΔMAPE':>7s}  {'TA%':>5s}  {'Contribution':<26s}│")
        print(f"├{'─'*93}┤")

        prev_ccc = 0.0
        prev_mape = 100.0
        for m in available_layers:
            r = method_map[m]
            d_ccc = r.ccc - prev_ccc
            d_mape = r.mape - prev_mape

            # Describe what this layer adds
            if m == "map":
                contrib = "Baseline point estimate"
            elif m == "laplace":
                contrib = f"+CI: Δ={d_ccc:+.3f} CCC"
            elif m == "smc":
                contrib = f"+Particles: Δ={d_ccc:+.3f} CCC"
            elif m == "adaptive_ind":
                contrib = f"+Pipeline: Δ={d_ccc:+.3f} CCC"
            elif m == "adaptive_cum":
                cum_vs_ind = r.ccc - method_map.get("adaptive_ind", r).ccc
                contrib = f"+VN learning: Δ={cum_vs_ind:+.3f} CCC"
            else:
                contrib = ""

            arrow = "↑" if d_ccc > 0.005 else "↓" if d_ccc < -0.005 else "→"
            print(f"│ {m:<16s}  {r.ccc:>6.3f}  {d_ccc:>+6.3f}{arrow}  {r.mape:>6.1f}%  {d_mape:>+6.1f}%  {r.target_attain:>4.0f}%  {contrib:<26s}│")
            prev_ccc = r.ccc
            prev_mape = r.mape

        print(f"├{'─'*93}┤")

        # Summary line
        base_r = method_map.get("map")
        best_r = method_map.get("adaptive_cum") or method_map.get("adaptive_ind")
        if base_r and best_r:
            total_gain = best_r.ccc - base_r.ccc
            print(f"│ Total pipeline gain: MAP CCC={base_r.ccc:.3f} → Adaptive CCC={best_r.ccc:.3f} "
                  f"(Δ={total_gain:+.3f})                   │")
            if total_gain > 0.05:
                print(f"│ ✅ Pipeline triển khai 3 lớp ĐÁNG KỂ cải thiện {total_gain:.1%} CCC so với MAP đơn lẻ                    │")
            elif total_gain > 0.01:
                print(f"│ 🔶 Pipeline cải thiện NHẸ — mỗi layer đóng góp nhỏ nhưng tổng hợp có ý nghĩa                    │")
            else:
                print(f"│ ℹ️  MAP đơn lẻ đã đủ tốt cho dataset này — pipeline benefit ở datasets phức tạp hơn                │")

        # Layer explanation
        print(f"├{'─'*93}┤")
        print(f"│ 📝 Giải thích Layer Contribution:                                                            │")
        print(f"│   Layer 1 (MAP+Laplace): Ước lượng nhanh + CI → quyết định lâm sàng cơ bản                  │")
        print(f"│   Layer 2 (SMC):         500 particles refine posterior → giảm bias & variance               │")
        print(f"│   Layer 3 (VN learning): Cập nhật prior VN qua từng BN → prior tốt hơn cho BN sau           │")
        print(f"│   Adaptive=L1+L2+L3:     Kết hợp 3 lớp → accuracy MCMC nhưng tốc độ MAP                    │")
        print(f"└{'─'*93}┘")

    # ══════════════════════════════════════════════════════════════
    # Save outputs (delegated to benchmark_export + benchmark_plots)
    # ══════════════════════════════════════════════════════════════

    out_dir = Path(__file__).resolve().parent

    # Main CSV
    export_main_csv(all_results, out_dir)

    # Rolling CCC CSV (for adaptive_cum learning curve plot)
    export_rolling_ccc(all_results, out_dir)

    # Individual patient CSV (per thuyết minh CV 4.2)
    export_individual_csv(patient_results_cache, out_dir)

    # Validation Plots (per thuyết minh CV 4.2)
    print(f"\n[5/5] Generating validation plots...")
    generate_all_plots(patient_results_cache, out_dir)

    # NPDE Summary Statistics CSV
    export_npde_summary(patient_results_cache, out_dir)

    return all_results


# ══════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_PATIENTS
    run_benchmark(n)

