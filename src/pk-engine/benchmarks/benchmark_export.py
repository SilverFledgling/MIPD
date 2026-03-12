"""
Benchmark Export — Save benchmark results to CSV files.

Single Responsibility: export data to CSV + NPDE summary stats.
Per thuyết minh Công việc 4.2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats


def export_main_csv(all_results: list, output_dir: Path) -> Path:
    """Export main benchmark summary CSV."""
    csv_path = output_dir / "benchmark_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "method,mpe,mape,rmse,ccc,"
            "ba_bias,ba_loa_lower,ba_loa_upper,"
            "coverage_95,target_attain,shrinkage_cl,"
            "tost_p,tost_equivalent,"
            "runtime_s,n_converged,n_failed,speed_per_patient\n"
        )
        for r in all_results:
            spd = r.runtime_s / max(r.n_converged, 1)
            cov = r.coverage_95 if not np.isnan(r.coverage_95) else -1
            shr = r.shrinkage_cl if not np.isnan(r.shrinkage_cl) else -1
            tp = r.tost_p if not np.isnan(r.tost_p) else -1
            f.write(
                f"{r.method},{r.mpe:.4f},{r.mape:.4f},{r.rmse:.4f},{r.ccc:.4f},"
                f"{r.ba_bias:.4f},{r.ba_loa_lower:.4f},{r.ba_loa_upper:.4f},"
                f"{cov:.2f},{r.target_attain:.2f},{shr:.2f},"
                f"{tp:.4f},{r.tost_equivalent},"
                f"{r.runtime_s:.2f},{r.n_converged},{r.n_failed},{spd:.3f}\n"
            )
    print(f"\n📊 CSV saved: {csv_path}")
    return csv_path


def export_rolling_ccc(all_results: list, output_dir: Path) -> None:
    """Export rolling CCC CSV for adaptive methods."""
    for r in all_results:
        if r.rolling_ccc is not None:
            rolling_path = output_dir / f"rolling_ccc_{r.method}.csv"
            with open(rolling_path, "w", encoding="utf-8") as f:
                f.write("patient_index,rolling_ccc\n")
                for i, v in enumerate(r.rolling_ccc):
                    val = f"{v:.6f}" if not np.isnan(v) else ""
                    f.write(f"{i + 1},{val}\n")
            print(f"📈 Rolling CCC saved: {rolling_path}")


def export_individual_csv(
    patient_results_cache: dict[str, list],
    output_dir: Path,
) -> Path:
    """Export individual patient results CSV."""
    csv_path = output_dir / "benchmark_individual.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("method,patient_id,cl_true,cl_est,pe_pct,ci_lower,ci_upper\n")
        for method, prs in patient_results_cache.items():
            for pr in prs:
                ci_lo = pr.ci_lower if pr.ci_lower is not None else ""
                ci_up = pr.ci_upper if pr.ci_upper is not None else ""
                pe = (pr.cl_est - pr.cl_true) / pr.cl_true * 100
                f.write(f"{method},{pr.cl_true:.4f},{pr.cl_true:.4f},"
                        f"{pr.cl_est:.4f},{pe:.2f},{ci_lo},{ci_up}\n")
    print(f"📋 Individual CSV saved: {csv_path}")
    return csv_path


def export_npde_summary(
    patient_results_cache: dict[str, list],
    output_dir: Path,
) -> Path:
    """Export NPDE test statistics CSV."""
    csv_path = output_dir / "npde_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("method,npde_mean,npde_sd,t_test_p,bartlett_p,"
                "shapiro_p,mean_ok,var_ok,normal_ok\n")
        for method, prs in patient_results_cache.items():
            if len(prs) < 8:
                continue
            true_cls = np.array([pr.cl_true for pr in prs])
            est_cls = np.array([pr.cl_est for pr in prs])
            pe = est_cls - true_cls
            pe_sd = np.std(pe, ddof=1) if np.std(pe, ddof=1) > 0 else 1e-6
            npde = (pe - np.mean(pe)) / pe_sd

            # Statistical tests
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

            f.write(f"{method},{np.mean(npde):.4f},{np.std(npde, ddof=1):.4f},"
                    f"{t_p:.4f},{bartlett_p:.4f},{shap_p:.4f},"
                    f"{mean_ok},{var_ok},{normal_ok}\n")
    print(f"📊 NPDE summary saved: {csv_path}")
    return csv_path
