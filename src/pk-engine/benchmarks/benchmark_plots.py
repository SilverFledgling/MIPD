"""
Benchmark Plots — Generate validation plots for PK estimation.

Single Responsibility: generate Bland-Altman, VPC, NPDE plots.
Per thuyết minh Công việc 4.2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_bland_altman(
    true_cls: np.ndarray,
    est_cls: np.ndarray,
    method: str,
    output_dir: Path,
) -> None:
    """Generate Bland-Altman plot for one method."""
    n = len(true_cls)
    means = (true_cls + est_cls) / 2
    diffs = est_cls - true_cls
    ba_bias = float(np.mean(diffs))
    ba_sd = float(np.std(diffs, ddof=1))
    loa_upper = ba_bias + 1.96 * ba_sd
    loa_lower = ba_bias - 1.96 * ba_sd

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(means, diffs, alpha=0.5, s=20, color='#2563eb')
    ax.axhline(ba_bias, color='#ef4444', linewidth=1.5,
               label=f'Bias = {ba_bias:+.3f}')
    ax.axhline(loa_upper, color='#f59e0b', linestyle='--', linewidth=1,
               label=f'+1.96 SD = {loa_upper:+.3f}')
    ax.axhline(loa_lower, color='#f59e0b', linestyle='--', linewidth=1,
               label=f'−1.96 SD = {loa_lower:+.3f}')
    ax.axhline(0, color='#94a3b8', linewidth=0.5)
    ax.set_xlabel('Mean of True & Estimated CL (L/h)')
    ax.set_ylabel('Difference (Estimated − True) (L/h)')
    ax.set_title(f'Bland-Altman: {method} (n={n})')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"bland_altman_{method}.png", dpi=150)
    plt.close(fig)


def generate_vpc(
    true_cls: np.ndarray,
    est_cls: np.ndarray,
    method: str,
    output_dir: Path,
) -> None:
    """Generate Visual Predictive Check (VPC) plot for one method."""
    pctiles = [5, 25, 50, 75, 95]
    true_pct = np.percentile(true_cls, pctiles)
    est_pct = np.percentile(est_cls, pctiles)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram overlay
    bins = np.linspace(
        min(true_cls.min(), est_cls.min()) * 0.8,
        max(true_cls.max(), est_cls.max()) * 1.2,
        30,
    )
    axes[0].hist(true_cls, bins=bins, alpha=0.5, label='True CL',
                 color='#22c55e', edgecolor='#166534')
    axes[0].hist(est_cls, bins=bins, alpha=0.5, label=f'Estimated ({method})',
                 color='#3b82f6', edgecolor='#1e3a5f')
    axes[0].set_xlabel('CL (L/h)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'VPC Distribution: {method}')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Right: QQ percentile comparison
    axes[1].scatter(true_pct, est_pct, s=60, color='#2563eb', zorder=3)
    for i, p in enumerate(pctiles):
        axes[1].annotate(f'P{p}', (true_pct[i], est_pct[i]),
                         fontsize=8, ha='left', va='bottom')
    lim_lo = min(true_pct.min(), est_pct.min()) * 0.8
    lim_hi = max(true_pct.max(), est_pct.max()) * 1.2
    axes[1].plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--',
                 alpha=0.5, label='Perfect agreement')
    axes[1].set_xlabel('True CL Percentile (L/h)')
    axes[1].set_ylabel('Estimated CL Percentile (L/h)')
    axes[1].set_title(f'VPC Percentiles: {method}')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_aspect('equal', adjustable='box')

    fig.tight_layout()
    fig.savefig(output_dir / f"vpc_{method}.png", dpi=150)
    plt.close(fig)


def generate_npde(
    true_cls: np.ndarray,
    est_cls: np.ndarray,
    method: str,
    output_dir: Path,
) -> None:
    """Generate NPDE histogram + QQ plot for one method."""
    n = len(true_cls)
    pe = est_cls - true_cls
    pe_mean = np.mean(pe)
    pe_sd = np.std(pe, ddof=1) if np.std(pe, ddof=1) > 0 else 1e-6
    npde = (pe - pe_mean) / pe_sd

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram vs N(0,1)
    axes[0].hist(npde, bins=20, density=True, alpha=0.6,
                 color='#3b82f6', edgecolor='#1e3a5f', label='NPDE')
    x_range = np.linspace(-4, 4, 100)
    axes[0].plot(x_range, scipy_stats.norm.pdf(x_range), 'r-',
                 linewidth=2, label='N(0,1)')
    axes[0].set_xlabel('NPDE')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'NPDE Distribution: {method}')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Right: QQ plot
    theoretical_q = scipy_stats.norm.ppf(
        (np.arange(1, n + 1) - 0.5) / n
    )
    sorted_npde = np.sort(npde)
    axes[1].scatter(theoretical_q, sorted_npde, s=15,
                    alpha=0.6, color='#2563eb')
    axes[1].plot([-4, 4], [-4, 4], 'r--', linewidth=1.5, label='y = x')
    axes[1].set_xlabel('Theoretical Quantiles (N(0,1))')
    axes[1].set_ylabel('Sample Quantiles (NPDE)')
    axes[1].set_title(f'NPDE QQ Plot: {method}')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"npde_{method}.png", dpi=150)
    plt.close(fig)


def generate_all_plots(
    patient_results_cache: dict[str, list],
    output_dir: Path,
) -> None:
    """Generate all validation plots for all methods."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for method, prs in patient_results_cache.items():
        if len(prs) < 3:
            continue
        true_cls = np.array([pr.cl_true for pr in prs])
        est_cls = np.array([pr.cl_est for pr in prs])

        generate_bland_altman(true_cls, est_cls, method, plots_dir)
        generate_vpc(true_cls, est_cls, method, plots_dir)
        generate_npde(true_cls, est_cls, method, plots_dir)

    print(f"       📊 Bland-Altman plots saved to: {plots_dir}")
    print(f"       📊 VPC plots saved to: {plots_dir}")
    print(f"       📊 NPDE plots saved to: {plots_dir}")
