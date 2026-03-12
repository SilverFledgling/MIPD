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

# TDM sampling times
TDM_TIMES = [1.5, 12.0, 24.0]
DOSE_AMOUNT = 1000.0          # mg per dose
INFUSION_DURATION = 1.0        # hour
DAILY_DOSE = 2 * DOSE_AMOUNT  # 2000 mg/day (q12h regimen)

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
    """Generate N virtual patients with CLAMPED parameters to avoid ODE issues."""
    model = VANCOMYCIN_VN
    omega = np.array(model.omega_matrix, dtype=np.float64)
    patients = []

    for i in range(n):
        # Random covariates — realistic ranges
        age = rng.uniform(25, 75)
        weight = rng.uniform(45, 100)
        height = rng.uniform(150, 185)
        scr = rng.uniform(0.5, 3.0)
        gender = Gender.MALE if rng.random() > 0.5 else Gender.FEMALE

        patient_data = PatientData(
            age=age, weight=weight, height=height,
            gender=gender, serum_creatinine=scr,
        )

        tv_params = compute_vancomycin_tv(patient_data)

        # Sample eta with CLAMPING to avoid extreme values
        true_eta = rng.multivariate_normal(np.zeros(omega.shape[0]), omega)
        true_eta = np.clip(true_eta, -0.8, 0.8)

        true_params = apply_iiv(tv_params, true_eta)

        # Clamp PK params to physiological range
        true_params = PKParams(
            CL=np.clip(true_params.CL, 0.5, 15.0),
            V1=np.clip(true_params.V1, 5.0, 100.0),
            Q=np.clip(true_params.Q, 0.5, 20.0),
            V2=np.clip(true_params.V2, 5.0, 150.0),
        )

        doses = [
            DoseEvent(time=0.0, amount=DOSE_AMOUNT, duration=INFUSION_DURATION),
            DoseEvent(time=12.0, amount=DOSE_AMOUNT, duration=INFUSION_DURATION),
        ]

        # True concentrations
        try:
            true_concs = predict_concentrations(
                true_params, doses, TDM_TIMES, model.model_type,
            )
            true_concs = [float(c) for c in true_concs]
        except Exception:
            continue

        # Verify valid concentrations
        if any(c <= 0 or np.isnan(c) or np.isinf(c) for c in true_concs):
            continue

        # Add residual error — 1-2 TDM samples
        n_tdm = rng.choice([1, 2], p=[0.5, 0.5])
        tdm_indices = sorted(rng.choice(len(TDM_TIMES), size=n_tdm, replace=False))

        observations = []
        true_at_tdm = []
        for idx in tdm_indices:
            c_true = true_concs[idx]
            sigma_prop = model.error_model.sigma_prop
            sigma_add = model.error_model.sigma_add
            sd = np.sqrt((sigma_prop * c_true) ** 2 + sigma_add ** 2)
            c_obs = max(0.5, c_true + rng.normal(0, sd))
            observations.append(
                Observation(time=TDM_TIMES[idx], concentration=round(c_obs, 2))
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
    print(f"       → {len(patients)} patients generated (valid)")

    cls = [p.true_params.CL for p in patients]
    print(f"       CL range: {min(cls):.2f} – {max(cls):.2f} L/h "
          f"(mean {np.mean(cls):.2f})")

    aucs = [DAILY_DOSE / cl for cl in cls]
    in_target = sum(1 for a in aucs if AUC_TARGET_LOW <= a <= AUC_TARGET_HIGH)
    print(f"       AUC target (400-600): {in_target}/{len(patients)} "
          f"({in_target / len(patients) * 100:.0f}%) patients in range")

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

