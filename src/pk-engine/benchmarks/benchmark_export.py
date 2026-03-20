"""
Benchmark Export — Save benchmark results to CSV files + timestamped backup.

Single Responsibility: export data to CSV + NPDE summary stats + backup.
Per thuyết minh Công việc 4.2.

Backup structure:
    benchmarks/
      backup/
        YYYY-MM-DD_HH-MM-SS.mmm/   ← one folder per run, ms-precise
          run_metadata.json          ← times, config, per-method info
          benchmark_results.csv
          benchmark_individual.csv
          npde_summary.csv
          rolling_ccc_*.csv
          plots/                     (plots copied here if available)
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats


# ══════════════════════════════════════════════════════════════════
# Backup helpers
# ══════════════════════════════════════════════════════════════════

def create_backup_dir(benchmarks_dir: Path) -> tuple[Path, datetime]:
    """
    Create a timestamped backup directory under benchmarks_dir/backup/.

    The folder name is: YYYY-MM-DD_HH-MM-SS.mmm  (ms precision, local time).

    Returns
    -------
    backup_dir : Path  — the newly created directory
    run_start  : datetime — the exact moment the folder was created
    """
    run_start = datetime.now()
    ts = run_start.strftime("%Y-%m-%d_%H-%M-%S.") + f"{run_start.microsecond // 1000:03d}"
    backup_dir = benchmarks_dir / "backup" / ts
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir, run_start


def export_run_metadata(
    backup_dir: Path,
    *,
    run_start: datetime,
    run_end: datetime,
    n_patients: int,
    n_valid_patients: int,
    seed: int,
    methods_timing: dict[str, dict],  # {method: {start, end, runtime_s}}
    all_results: list,
    save_timestamp: datetime | None = None,
) -> Path:
    """
    Write run_metadata.json into backup_dir.

    Fields
    ------
    run_created_at   : ISO-8601 local time this backup was created
    run_start        : when run_benchmark() started (ms precision)
    run_end          : when run_benchmark() finished (ms precision)
    run_duration_s   : total wall-clock seconds
    save_timestamp   : when outputs were written to disk (ms precision)
    n_patients_requested : N passed to run_benchmark()
    n_patients_valid     : patients successfully generated
    seed
    methods          : list of method timing dicts with per-method metrics
    """
    if save_timestamp is None:
        save_timestamp = datetime.now()

    def _fmt(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"

    # Build per-method summary
    methods_meta = []
    results_by_method = {r.method: r for r in all_results}
    for method, timing in methods_timing.items():
        entry: dict = {
            "method": method,
            "start_time":  _fmt(timing["start"]),
            "end_time":    _fmt(timing["end"]),
            "runtime_s":   round(timing["runtime_s"], 3),
        }
        r = results_by_method.get(method)
        if r is not None:
            entry["n_converged"] = r.n_converged
            entry["n_failed"]    = r.n_failed
            entry["n_total"]     = r.n_total
            entry["mpe"]         = round(float(r.mpe),  4)
            entry["mape"]        = round(float(r.mape), 4)
            entry["rmse"]        = round(float(r.rmse), 4)
            entry["ccc"]         = round(float(r.ccc),  4)
            entry["target_attain"] = round(float(r.target_attain), 2)
        methods_meta.append(entry)

    meta = {
        "run_created_at":        _fmt(run_start),
        "run_start":             _fmt(run_start),
        "run_end":               _fmt(run_end),
        "run_duration_s":        round((run_end - run_start).total_seconds(), 3),
        "save_timestamp":        _fmt(save_timestamp),
        "n_patients_requested":  n_patients,
        "n_patients_valid":      n_valid_patients,
        "seed":                  seed,
        "backup_folder":         backup_dir.name,
        "methods":               methods_meta,
    }

    meta_path = backup_dir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"📋 Metadata saved: {meta_path}")
    return meta_path


# ══════════════════════════════════════════════════════════════════
# CSV export functions (unchanged API, dual-save: live + backup)
# ══════════════════════════════════════════════════════════════════

def export_main_csv(
    all_results: list,
    output_dir: Path,
    backup_dir: Path | None = None,
) -> Path:
    """Export main benchmark summary CSV."""
    content_lines = [
        "method,mpe,mape,rmse,ccc,"
        "ba_bias,ba_loa_lower,ba_loa_upper,"
        "coverage_95,target_attain,shrinkage_cl,"
        "tost_p,tost_equivalent,"
        "runtime_s,n_converged,n_failed,speed_per_patient\n"
    ]
    for r in all_results:
        spd = r.runtime_s / max(r.n_converged, 1)
        cov = r.coverage_95 if not np.isnan(r.coverage_95) else -1
        shr = r.shrinkage_cl if not np.isnan(r.shrinkage_cl) else -1
        tp = r.tost_p if not np.isnan(r.tost_p) else -1
        content_lines.append(
            f"{r.method},{r.mpe:.4f},{r.mape:.4f},{r.rmse:.4f},{r.ccc:.4f},"
            f"{r.ba_bias:.4f},{r.ba_loa_lower:.4f},{r.ba_loa_upper:.4f},"
            f"{cov:.2f},{r.target_attain:.2f},{shr:.2f},"
            f"{tp:.4f},{r.tost_equivalent},"
            f"{r.runtime_s:.2f},{r.n_converged},{r.n_failed},{spd:.3f}\n"
        )
    content = "".join(content_lines)

    csv_path = output_dir / "benchmark_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n📊 CSV saved: {csv_path}")

    if backup_dir is not None:
        bak_path = backup_dir / "benchmark_results.csv"
        with open(bak_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"   💾 Backup: {bak_path}")

    return csv_path


def export_rolling_ccc(
    all_results: list,
    output_dir: Path,
    backup_dir: Path | None = None,
) -> None:
    """Export rolling CCC CSV for adaptive methods."""
    for r in all_results:
        if r.rolling_ccc is not None:
            lines = ["patient_index,rolling_ccc\n"]
            for i, v in enumerate(r.rolling_ccc):
                val = f"{v:.6f}" if not np.isnan(v) else ""
                lines.append(f"{i + 1},{val}\n")
            content = "".join(lines)

            rolling_path = output_dir / f"rolling_ccc_{r.method}.csv"
            with open(rolling_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"📈 Rolling CCC saved: {rolling_path}")

            if backup_dir is not None:
                bak_path = backup_dir / f"rolling_ccc_{r.method}.csv"
                with open(bak_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"   💾 Backup: {bak_path}")


def export_individual_csv(
    patient_results_cache: dict[str, list],
    output_dir: Path,
    backup_dir: Path | None = None,
) -> Path:
    """Export individual patient results CSV."""
    lines = ["method,patient_id,cl_true,cl_est,pe_pct,ci_lower,ci_upper\n"]
    for method, prs in patient_results_cache.items():
        for pr in prs:
            ci_lo = pr.ci_lower if pr.ci_lower is not None else ""
            ci_up = pr.ci_upper if pr.ci_upper is not None else ""
            pe = (pr.cl_est - pr.cl_true) / pr.cl_true * 100
            lines.append(
                f"{method},{pr.cl_true:.4f},{pr.cl_true:.4f},"
                f"{pr.cl_est:.4f},{pe:.2f},{ci_lo},{ci_up}\n"
            )
    content = "".join(lines)

    csv_path = output_dir / "benchmark_individual.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📋 Individual CSV saved: {csv_path}")

    if backup_dir is not None:
        bak_path = backup_dir / "benchmark_individual.csv"
        with open(bak_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"   💾 Backup: {bak_path}")

    return csv_path


def export_npde_summary(
    patient_results_cache: dict[str, list],
    output_dir: Path,
    backup_dir: Path | None = None,
) -> Path:
    """Export NPDE test statistics CSV."""
    lines = [
        "method,npde_mean,npde_sd,t_test_p,bartlett_p,"
        "shapiro_p,mean_ok,var_ok,normal_ok\n"
    ]
    for method, prs in patient_results_cache.items():
        if len(prs) < 8:
            continue
        true_cls = np.array([pr.cl_true for pr in prs])
        est_cls = np.array([pr.cl_est for pr in prs])
        pe = est_cls - true_cls
        pe_sd = np.std(pe, ddof=1) if np.std(pe, ddof=1) > 0 else 1e-6
        npde = (pe - np.mean(pe)) / pe_sd

        t_stat, t_p = scipy_stats.ttest_1samp(npde, 0)
        chi2_stat = (len(npde) - 1) * np.var(npde, ddof=1)
        bartlett_p = scipy_stats.chi2.sf(chi2_stat, len(npde) - 1) * 2
        bartlett_p = min(bartlett_p, 1.0)
        if len(npde) >= 8:
            shap_stat, shap_p = scipy_stats.shapiro(npde[:5000])
        else:
            shap_p = np.nan

        mean_ok = "YES" if t_p > 0.05 else "NO"
        var_ok = "YES" if bartlett_p > 0.05 else "NO"
        normal_ok = "YES" if (not np.isnan(shap_p) and shap_p > 0.05) else "NO"

        lines.append(
            f"{method},{np.mean(npde):.4f},{np.std(npde, ddof=1):.4f},"
            f"{t_p:.4f},{bartlett_p:.4f},{shap_p:.4f},"
            f"{mean_ok},{var_ok},{normal_ok}\n"
        )
    content = "".join(lines)

    csv_path = output_dir / "npde_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📊 NPDE summary saved: {csv_path}")

    if backup_dir is not None:
        bak_path = backup_dir / "npde_summary.csv"
        with open(bak_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"   💾 Backup: {bak_path}")

    return csv_path


def copy_plots_to_backup(output_dir: Path, backup_dir: Path) -> None:
    """Copy any generated plots into backup_dir/plots/."""
    plots_src = output_dir / "plots"
    if plots_src.exists() and any(plots_src.iterdir()):
        plots_dst = backup_dir / "plots"
        if plots_dst.exists():
            shutil.rmtree(plots_dst)
        shutil.copytree(plots_src, plots_dst)
        n = sum(1 for _ in plots_dst.rglob("*") if _.is_file())
        print(f"   📁 Plots backed up: {plots_dst} ({n} files)")
