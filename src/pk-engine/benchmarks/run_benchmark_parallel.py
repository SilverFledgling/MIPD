"""
Parallel Benchmark Runner — Runs selected methods with multiprocessing.

Usage:
    cd src/pk-engine

    # CMD 1: Fast + medium methods
    python benchmarks/run_benchmark_parallel.py --methods map,laplace,smc,ep,mcmc_mh --n 1000 --workers 3

    # CMD 2: Advanced methods
    python benchmarks/run_benchmark_parallel.py --methods mcmc_nuts,advi,adaptive_ind,adaptive_cum --n 1000 --workers 3

    # Single method test
    python benchmarks/run_benchmark_parallel.py --methods ep --n 20 --workers 1

Notes:
    - 8GB RAM: use --workers 3 (safe, ~400MB/worker)
    - adaptive_cum runs SEQUENTIALLY regardless of --workers (feedback loop)
    - Results saved to benchmarks/results_{methods_hash}.csv
    - Use merge_results.py to combine CSVs from multiple CMDs

Reference:
    Roberts GO, Rosenthal JS (2009). J Appl Prob — Adaptive MCMC
    Rowland & Tozer (2011). Clinical PK/PD — Superposition principle
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Ensure pk-engine is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_bench_dir = str(Path(__file__).resolve().parent)
if _bench_dir not in sys.path:
    sys.path.insert(0, _bench_dir)

from pk.models import DoseEvent, Observation, PKParams, ModelType, PatientData, Gender
from pk.population import VANCOMYCIN_VN, compute_vancomycin_tv, apply_iiv
from pk.analytical import predict_analytical


# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

SEED = 42
DAILY_DOSE = 2000.0
AUC_TARGET_LOW = 400.0
AUC_TARGET_HIGH = 600.0
INFUSION_DURATION = 1.0

OMEGA_PRIOR_DIAG = np.array([
    VANCOMYCIN_VN.omega_matrix[i][i]
    for i in range(len(VANCOMYCIN_VN.omega_matrix))
])


# ══════════════════════════════════════════════════════════════════
# Data containers (same as run_benchmark.py)
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
    age: float = 0.0
    weight: float = 0.0
    height: float = 0.0
    scr: float = 0.0
    crcl: float = 0.0
    gender: str = "M"


@dataclass
class PatientResult:
    cl_true: float
    cl_est: float
    eta_est: list[float] | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    converged: bool = True


# ══════════════════════════════════════════════════════════════════
# Patient Generation — Gaussian Copula (NORTA)
# ══════════════════════════════════════════════════════════════════

def generate_virtual_patients_copula(n: int, rng: np.random.Generator) -> list[PatientSim]:
    """
    Generate N virtual patients using Gaussian Copula (NORTA method).

    NORTA = NORmal-To-Anything: sample correlated normals, then transform
    to target marginals via inverse CDF.

    References:
        Smania & Jonsson (2021), CPT Pharmacometrics — Copula in pharmacometrics
        Cario & Nelson (1997) — NORTA method
        GSO Vietnam (2023) — Vietnamese demographics
        Hien et al. (2021), Kidney Int Rep — SCr in VN
        Rybak et al. (2020), AJHP — Vancomycin dosing guidelines
    """
    from scipy.stats import norm, lognorm
    from pk.clinical import cockcroft_gault_crcl

    model = VANCOMYCIN_VN
    omega = np.array(model.omega_matrix, dtype=np.float64)
    patients = []

    # ── Vietnamese population parameters (gender-specific) ──
    MALE_RATIO = 0.55

    # Correlation matrix: [Age, Weight, Height, ln(SCr)]
    CORR_MATRIX = np.array([
        [1.000, -0.10,  0.00,  0.25],   # Age
        [-0.10, 1.000,  0.60,  0.05],   # WT
        [0.00,  0.60,  1.000, -0.05],   # HT
        [0.25,  0.05, -0.05,  1.000],   # ln(SCr)
    ])
    L = np.linalg.cholesky(CORR_MATRIX)

    # Marginal parameters (gender-specific)
    params_male = {
        'age_mean': 52.0, 'age_sd': 15.0,
        'wt_mean': 62.0, 'wt_sd': 10.0,
        'ht_mean': 165.0, 'ht_sd': 6.0,
        'scr_log_mean': np.log(0.95), 'scr_log_sd': 0.30,
    }
    params_female = {
        'age_mean': 56.0, 'age_sd': 16.0,
        'wt_mean': 52.0, 'wt_sd': 8.0,
        'ht_mean': 155.0, 'ht_sd': 5.0,
        'scr_log_mean': np.log(0.75), 'scr_log_sd': 0.28,
    }

    DOSE_PER_KG = 15.0  # mg/kg (IDSA/ASHP 2020)

    for i in range(n):
        is_male = rng.random() < MALE_RATIO
        gender = Gender.MALE if is_male else Gender.FEMALE
        p = params_male if is_male else params_female

        # NORTA: correlated standard normals → transform to marginals
        z = rng.standard_normal(4)
        z_corr = L @ z

        # Transform: z → uniform via Φ(z) → target marginals via F^{-1}
        u = norm.cdf(z_corr)  # Uniform on [0, 1]

        # Age: Truncated Normal → use raw normal + clip
        age = p['age_mean'] + p['age_sd'] * z_corr[0]
        age = np.clip(age, 18.0, 95.0)

        # Weight: Normal
        weight = p['wt_mean'] + p['wt_sd'] * z_corr[1]
        weight = np.clip(weight, 30.0, 130.0)

        # Height: Normal
        height = p['ht_mean'] + p['ht_sd'] * z_corr[2]
        height = np.clip(height, 140.0, 195.0)

        # SCr: LogNormal (correlated via copula)
        scr = np.exp(p['scr_log_mean'] + p['scr_log_sd'] * z_corr[3])
        scr = np.clip(scr, 0.3, 8.0)

        # ── CKD-by-Age adjustment (Hien et al. 2021, Kidney Int Rep) ──
        # Prevalence of CKD (eGFR < 60, Stage 3-5) by age in Vietnam:
        #   18-39: ~1%  |  40-59: ~3%  |  60-79: ~8%  |  ≥80: ~15%
        if age < 40:
            p_ckd_moderate = 0.01
            p_ckd_severe   = 0.005
        elif age < 60:
            p_ckd_moderate = 0.03
            p_ckd_severe   = 0.01
        elif age < 80:
            p_ckd_moderate = 0.08
            p_ckd_severe   = 0.03
        else:
            p_ckd_moderate = 0.15
            p_ckd_severe   = 0.06

        ckd_roll = rng.random()
        if ckd_roll < p_ckd_severe:
            scr = rng.uniform(3.0, 6.0)       # CKD Stage 4-5
        elif ckd_roll < p_ckd_moderate + p_ckd_severe:
            scr = rng.uniform(1.5, 3.0)        # CKD Stage 3

        patient_data = PatientData(
            age=age, weight=weight, height=height,
            gender=gender, serum_creatinine=scr,
        )
        tv_params = compute_vancomycin_tv(patient_data)

        # IIV: η ~ MVN(0, Ω)
        true_eta = rng.multivariate_normal(np.zeros(omega.shape[0]), omega)
        true_eta = np.clip(true_eta, -2.5, 2.5)
        true_params = apply_iiv(tv_params, true_eta)

        # Physiological clamps
        true_params = PKParams(
            CL=np.clip(true_params.CL, 0.3, 20.0),
            V1=np.clip(true_params.V1, 3.0, 120.0),
            Q=np.clip(true_params.Q, 0.2, 25.0),
            V2=np.clip(true_params.V2, 3.0, 200.0),
        )

        # Dosing: 15 mg/kg rounded to 250mg
        single_dose = round(DOSE_PER_KG * weight / 250.0) * 250.0
        single_dose = np.clip(single_dose, 500.0, 3000.0)

        # Interval based on CrCL
        crcl_val = cockcroft_gault_crcl(age, weight, scr, gender)
        if crcl_val > 80:
            interval, n_doses = 8.0, 6
        elif crcl_val > 50:
            interval, n_doses = 12.0, 4
        else:
            interval, n_doses = 24.0, 3

        doses = [
            DoseEvent(time=d * interval, amount=single_dose, duration=INFUSION_DURATION)
            for d in range(n_doses)
        ]

        # TDM times
        last_dose_time = doses[-1].time
        tdm_peak = last_dose_time + 1.5
        tdm_trough = last_dose_time + interval - 0.5
        tdm_mid = last_dose_time + interval / 2
        TDM_TIMES = [tdm_peak, tdm_mid, tdm_trough]

        # True concentrations (analytical!)
        try:
            true_concs = predict_analytical(true_params, doses, TDM_TIMES, model.model_type)
            true_concs = [float(c) for c in true_concs]
        except Exception:
            continue

        if any(c <= 0 or np.isnan(c) or np.isinf(c) for c in true_concs):
            continue

        # TDM observations with residual error
        n_tdm = rng.choice([1, 2], p=[0.4, 0.6])
        tdm_indices = sorted(rng.choice(len(TDM_TIMES), size=n_tdm, replace=False))

        observations = []
        true_at_tdm = []
        for idx in tdm_indices:
            c_true = true_concs[idx]
            sigma_prop = model.error_model.sigma_prop
            sigma_add = model.error_model.sigma_add
            sd = np.sqrt((sigma_prop * c_true) ** 2 + sigma_add ** 2)
            c_obs = max(0.5, c_true + rng.normal(0, sd))

            # Timing noise ±15 min
            t_noise = rng.uniform(-0.25, 0.25)
            t_actual = max(0.1, TDM_TIMES[idx] + t_noise)

            observations.append(Observation(time=t_actual, concentration=c_obs))
            true_at_tdm.append(c_true)

        patients.append(PatientSim(
            patient_id=len(patients) + 1,
            true_params=true_params,
            true_eta=true_eta,
            tv_params=tv_params,
            doses=doses,
            observations=observations,
            true_concentrations=true_at_tdm,
            age=age, weight=weight, height=height,
            scr=scr, crcl=crcl_val,
            gender="M" if is_male else "F",
        ))

    return patients


# ══════════════════════════════════════════════════════════════════
# Per-Patient Estimation (same logic as run_benchmark.py)
# ══════════════════════════════════════════════════════════════════

def estimate_for_patient(method: str, patient: PatientSim) -> PatientResult | None:
    """Run one Bayesian method on one patient."""
    from bayesian.engine import estimate

    model = VANCOMYCIN_VN
    engine_method = method
    kwargs = {}

    if method == "adaptive_ind":
        engine_method = "adaptive"
        try:
            from bayesian.population_store import VietnamPopulationStore
            VietnamPopulationStore.reset_instance()
        except Exception:
            pass
        kwargs = {"run_layer2": True, "run_layer3": False, "smc_n_particles": 500}
    elif method == "adaptive_cum":
        engine_method = "adaptive"
        kwargs = {"run_layer2": True, "run_layer3": False, "smc_n_particles": 500}
    elif engine_method in ("mcmc", "mcmc_nuts"):
        kwargs = {"n_warmup": 1000, "n_samples": 2000, "n_chains": 2}
    elif engine_method == "mcmc_mh":
        kwargs = {"n_warmup": 500, "n_samples": 1000, "n_chains": 2}

    result = estimate(
        method=engine_method, model=model,
        tv_params=patient.tv_params,
        doses=patient.doses,
        observations=patient.observations,
        **kwargs,
    )

    if result is None:
        return None

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

    # CI extraction
    ci_lower = ci_upper = None
    confidence = None
    if is_adaptive:
        confidence = getattr(result, 'final_confidence', None)
    if confidence is None:
        confidence = getattr(result, 'confidence', None)

    if confidence and isinstance(confidence, dict) and 'CL' in confidence:
        cl_ci = confidence['CL']
        ci_lower = cl_ci.get('ci95_lower')
        ci_upper = cl_ci.get('ci95_upper')

    if ci_lower is None or ci_upper is None:
        omega_cl = float(OMEGA_PRIOR_DIAG[0]) if len(OMEGA_PRIOR_DIAG) > 0 else 0.25
        ci_lower = cl_est * np.exp(-1.96 * np.sqrt(omega_cl))
        ci_upper = cl_est * np.exp(+1.96 * np.sqrt(omega_cl))

    # Eta
    eta_est = None
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


def _worker_estimate(args):
    """Worker function for multiprocessing Pool."""
    method, patient_data = args
    try:
        result = estimate_for_patient(method, patient_data)
        return result
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════════
# Metrics (simplified — reuse from benchmark_metrics)
# ══════════════════════════════════════════════════════════════════

def compute_metrics_simple(results: list[PatientResult]) -> dict:
    """Compute core metrics."""
    from benchmark_metrics import compute_metrics
    return compute_metrics(
        results,
        daily_dose=DAILY_DOSE,
        auc_target_low=AUC_TARGET_LOW,
        auc_target_high=AUC_TARGET_HIGH,
        omega_prior_diag=OMEGA_PRIOR_DIAG,
    )


# ══════════════════════════════════════════════════════════════════
# Main Parallel Runner
# ══════════════════════════════════════════════════════════════════

def run_parallel(methods: list[str], n_patients: int, n_workers: int,
                 load_patients_file: str | None = None):
    """Run benchmark methods with optional parallelism."""
    from datetime import datetime as _dt

    print("═" * 80)
    print(f"  MIPD Parallel Benchmark — {n_patients} patients × {len(methods)} methods")
    print(f"  Workers: {n_workers} | RAM-safe for 8GB")
    if load_patients_file:
        print(f"  Patients: loaded from {load_patients_file}")
    print(f"  Start: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 80)

    rng = np.random.default_rng(SEED)

    # Generate or load patients
    pkl_path = Path(__file__).resolve().parent / f"patients_{n_patients}.pkl"
    if load_patients_file:
        load_path = Path(load_patients_file)
        if not load_path.is_absolute():
            load_path = Path(__file__).resolve().parent / load_path
        print(f"\n[1/3] Loading patients from {load_path.name}...")
        with open(load_path, 'rb') as f:
            patients = pickle.load(f)
        print(f"       → {len(patients)} patients loaded")
    else:
        print(f"\n[1/3] Generating {n_patients} virtual patients (Copula)...")
        patients = generate_virtual_patients_copula(n_patients, rng)
        print(f"       → {len(patients)} valid patients")
        # Save for reuse by other CMDs
        with open(pkl_path, 'wb') as f:
            pickle.dump(patients, f)
        print(f"       → Saved to {pkl_path.name}")

    # Run methods
    print(f"\n[2/3] Running estimation methods...\n")
    all_csv_rows = []

    for method in methods:
        print(f"  ▸ {method:16s}  ", end="", flush=True)

        t_start = time.time()

        if method == "adaptive_cum":
            # Sequential — must maintain state across patients
            try:
                from bayesian.population_store import VietnamPopulationStore
                VietnamPopulationStore.reset_instance()
            except Exception:
                pass

            patient_results = []
            n_failed = 0
            for pi, p in enumerate(patients):
                if pi > 0 and pi % 50 == 0:
                    print(f"{pi}", end=".", flush=True)
                try:
                    pr = estimate_for_patient(method, p)
                    if pr is not None:
                        patient_results.append(pr)
                    else:
                        n_failed += 1
                except Exception:
                    n_failed += 1

        elif n_workers > 1:
            # Parallel — use multiprocessing Pool
            tasks = [(method, p) for p in patients]

            with multiprocessing.Pool(n_workers) as pool:
                results_raw = []
                for i, result in enumerate(pool.imap(_worker_estimate, tasks, chunksize=10)):
                    if (i + 1) % 50 == 0:
                        print(f"{i+1}", end=".", flush=True)
                    results_raw.append(result)

            patient_results = [r for r in results_raw if r is not None]
            n_failed = sum(1 for r in results_raw if r is None)

        else:
            # Sequential fallback
            patient_results = []
            n_failed = 0
            for pi, p in enumerate(patients):
                if pi > 0 and pi % 50 == 0:
                    print(f"{pi}", end=".", flush=True)
                try:
                    pr = estimate_for_patient(method, p)
                    if pr is not None:
                        patient_results.append(pr)
                    else:
                        n_failed += 1
                except Exception:
                    n_failed += 1

        t_elapsed = time.time() - t_start
        n_conv = len(patient_results)
        speed = t_elapsed / max(n_conv, 1)

        # Compute metrics
        if n_conv >= 3:
            metrics = compute_metrics_simple(patient_results)
        else:
            metrics = {k: 0.0 for k in [
                "mpe", "mape", "rmse", "ccc", "ba_bias",
                "ba_loa_lower", "ba_loa_upper", "coverage_95",
                "target_attain", "shrinkage_cl", "tost_p",
            ]}
            metrics["tost_equivalent"] = False

        print(f"  done ({t_elapsed:.1f}s, {speed:.2f}s/pt, CCC={metrics['ccc']:.3f})")

        # CSV row
        all_csv_rows.append({
            "method": method,
            "mpe": f"{metrics['mpe']:.4f}",
            "mape": f"{metrics['mape']:.4f}",
            "rmse": f"{metrics['rmse']:.4f}",
            "ccc": f"{metrics['ccc']:.4f}",
            "ba_bias": f"{metrics['ba_bias']:.4f}",
            "ba_loa_lower": f"{metrics['ba_loa_lower']:.4f}",
            "ba_loa_upper": f"{metrics['ba_loa_upper']:.4f}",
            "coverage_95": f"{metrics['coverage_95']:.2f}",
            "target_attain": f"{metrics['target_attain']:.2f}",
            "shrinkage_cl": f"{metrics['shrinkage_cl']:.2f}",
            "tost_p": f"{metrics['tost_p']:.4f}",
            "tost_equivalent": str(metrics.get('tost_equivalent', False)),
            "runtime_s": f"{t_elapsed:.2f}",
            "n_converged": n_conv,
            "n_failed": n_failed,
            "speed_per_patient": f"{speed:.3f}",
        })

    # Save CSV — filename includes timestamp (seconds since epoch)
    ts = int(time.time())
    methods_tag = "_".join(methods[:3])
    csv_path = Path(__file__).resolve().parent / f"results_{methods_tag}_{ts}.csv"
    fieldnames = list(all_csv_rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\n[3/3] Results saved: {csv_path.name}")

    # Summary table
    print(f"\n{'═'*80}")
    print(f"  {'Method':<16s}  {'CCC':>6s}  {'MAPE':>7s}  {'RMSE':>6s}  {'Cover':>6s}  {'Time':>8s}  {'Speed':>8s}")
    print(f"  {'─'*16}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}")
    for row in all_csv_rows:
        print(f"  {row['method']:<16s}  {row['ccc']:>6s}  {row['mape']:>6s}%  {row['rmse']:>6s}  "
              f"{row['coverage_95']:>5s}%  {row['runtime_s']:>7s}s  {row['speed_per_patient']:>7s}s")
    print(f"{'═'*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIPD Parallel Benchmark Runner")
    parser.add_argument("--methods", type=str, required=True,
                        help="Comma-separated method list (e.g., map,laplace,smc)")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of virtual patients (default: 50)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Multiprocessing workers (default: 3, safe for 8GB RAM)")
    parser.add_argument("--load-patients", type=str, default=None,
                        help="Load patients from pkl file (e.g., patients_50.pkl). "
                             "If not set, generates new patients and saves to pkl.")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    run_parallel(methods, args.n, args.workers, load_patients_file=args.load_patients)
