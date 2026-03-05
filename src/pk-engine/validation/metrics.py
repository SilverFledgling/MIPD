"""
Validation Metrics – Statistical metrics for model validation.

Implements:
    - MPE  (Median Prediction Error)
    - MAPE (Median Absolute Prediction Error)
    - RMSE (Root Mean Squared Error)
    - CCC  (Concordance Correlation Coefficient)
    - Bias and Precision analysis
    - Bland-Altman Limits of Agreement
    - Coverage probability (95% CI)
    - TOST equivalence test

Reference:
    - Lin (1989), Biometrics 45(1), 255-268 (CCC)
    - Bland & Altman (1986), Lancet 1(8476), 307-310
    - Schuirmann (1987), J Pharmacokinet Biopharm 15(6), 657-680 (TOST)

Dependencies: numpy, scipy
"""

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────

@dataclass
class PredictionMetrics:
    """
    Comprehensive prediction accuracy metrics.

    All metrics compare estimated vs true (or observed) values.
    """
    n: int

    # Bias (accuracy)
    mpe: float           # Median Prediction Error (%)
    mean_pe: float       # Mean Prediction Error (%)

    # Precision
    mape: float          # Median Absolute Prediction Error (%)
    mean_ape: float      # Mean Absolute Prediction Error (%)
    rmse: float          # Root Mean Squared Error

    # Agreement
    ccc: float           # Concordance Correlation Coefficient
    pearson_r: float     # Pearson correlation

    # Bland-Altman
    bias: float          # Mean difference (estimated - true)
    loa_lower: float     # Lower Limit of Agreement (bias - 1.96*SD)
    loa_upper: float     # Upper Limit of Agreement (bias + 1.96*SD)


@dataclass
class TOSTResult:
    """Result of Two One-Sided Tests for equivalence."""
    margin: float           # Equivalence margin (fraction, e.g. 0.10 = 10%)
    mean_diff: float        # Mean PE (%)
    se: float               # Standard error
    t1: float               # t-statistic for upper test
    t2: float               # t-statistic for lower test
    p1: float               # p-value for upper test
    p2: float               # p-value for lower test
    p_tost: float           # max(p1, p2)
    is_equivalent: bool     # p_TOST < 0.05
    ci90_lower: float       # 90% CI lower bound
    ci90_upper: float       # 90% CI upper bound


@dataclass
class NPDEResult:
    """Result of Normalized Prediction Distribution Errors analysis."""
    npde_values: NDArray[np.float64]
    mean: float
    variance: float
    mean_test_p: float          # t-test p-value (H0: mean=0)
    variance_test_p: float      # Fisher test p-value (H0: var=1)
    normality_test_p: float     # Shapiro-Wilk p-value
    is_adequate: bool           # All p > 0.05


# ──────────────────────────────────────────────────────────────────
# Prediction Error calculations
# ──────────────────────────────────────────────────────────────────

def prediction_errors(
    estimated: NDArray[np.float64],
    true_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute percentage prediction errors.

    PE_i = (estimated_i - true_i) / true_i * 100%
    """
    mask = true_values > 0
    pe = np.full_like(estimated, np.nan)
    pe[mask] = (estimated[mask] - true_values[mask]) / true_values[mask] * 100.0
    return pe


def compute_metrics(
    estimated: NDArray[np.float64],
    true_values: NDArray[np.float64],
) -> PredictionMetrics:
    """
    Compute comprehensive prediction accuracy metrics.

    Args:
        estimated:   Estimated/predicted values
        true_values: True/reference values

    Returns:
        PredictionMetrics with all metrics
    """
    n = len(estimated)
    if n != len(true_values):
        raise ValueError("Arrays must have same length")
    if n < 2:
        raise ValueError("Need at least 2 data points")

    # Prediction errors (%)
    pe = prediction_errors(estimated, true_values)
    valid = ~np.isnan(pe)
    pe_valid = pe[valid]

    # Bias metrics
    mpe = float(np.median(pe_valid))
    mean_pe = float(np.mean(pe_valid))

    # Precision metrics
    ape = np.abs(pe_valid)
    mape = float(np.median(ape))
    mean_ape = float(np.mean(ape))

    # RMSE
    diff = estimated - true_values
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # CCC (Lin's Concordance Correlation Coefficient)
    ccc = concordance_correlation(estimated, true_values)

    # Pearson r
    if np.std(estimated) > 0 and np.std(true_values) > 0:
        pearson_r = float(np.corrcoef(estimated, true_values)[0, 1])
    else:
        pearson_r = 0.0

    # Bland-Altman
    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    loa_lower = bias - 1.96 * sd_diff
    loa_upper = bias + 1.96 * sd_diff

    return PredictionMetrics(
        n=n,
        mpe=mpe,
        mean_pe=mean_pe,
        mape=mape,
        mean_ape=mean_ape,
        rmse=rmse,
        ccc=ccc,
        pearson_r=pearson_r,
        bias=bias,
        loa_lower=loa_lower,
        loa_upper=loa_upper,
    )


# ──────────────────────────────────────────────────────────────────
# CCC (Lin 1989)
# ──────────────────────────────────────────────────────────────────

def concordance_correlation(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """
    Lin's Concordance Correlation Coefficient.

    CCC = 2 * rho * sigma_x * sigma_y /
          (sigma_x^2 + sigma_y^2 + (mu_x - mu_y)^2)

    Where rho = Pearson correlation coefficient.

    Args:
        x: First set of measurements
        y: Second set of measurements

    Returns:
        CCC value in [-1, 1]
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    var_x = np.var(x, ddof=0)
    var_y = np.var(y, ddof=0)

    if var_x == 0 and var_y == 0:
        return 1.0 if abs(mu_x - mu_y) < 1e-10 else 0.0

    cov_xy = np.mean((x - mu_x) * (y - mu_y))
    denominator = var_x + var_y + (mu_x - mu_y) ** 2

    if denominator == 0:
        return 0.0

    return float(2.0 * cov_xy / denominator)


# ──────────────────────────────────────────────────────────────────
# TOST (Schuirmann 1987)
# ──────────────────────────────────────────────────────────────────

def tost_equivalence(
    estimated: NDArray[np.float64],
    true_values: NDArray[np.float64],
    margin: float = 0.10,
) -> TOSTResult:
    """
    Two One-Sided Tests for equivalence.

    H0: |PE| >= margin  vs  H1: |PE| < margin
    Equivalent if p_TOST < 0.05, or 90% CI within (-margin, +margin).

    Args:
        estimated:   Estimated values
        true_values: True values
        margin:      Equivalence margin (default 10%)

    Returns:
        TOSTResult with test statistics and conclusion
    """
    pe = prediction_errors(estimated, true_values)
    valid = ~np.isnan(pe)
    pe_valid = pe[valid]
    n = len(pe_valid)

    mean_diff = float(np.mean(pe_valid))
    sd = float(np.std(pe_valid, ddof=1))
    se = sd / np.sqrt(n)

    margin_pct = margin * 100.0  # Convert to percentage

    # Two one-sided t-tests (Schuirmann 1987)
    # Test 1: H0: mean_diff >= +margin  → reject if t1 sufficiently negative
    # Test 2: H0: mean_diff <= -margin  → reject if t2 sufficiently positive
    t1 = (mean_diff - margin_pct) / se if se > 0 else float("-inf")
    t2 = (mean_diff + margin_pct) / se if se > 0 else float("inf")

    p1 = float(stats.t.cdf(t1, df=n - 1))            # Left-tail p-value
    p2 = 1.0 - float(stats.t.cdf(t2, df=n - 1))      # Right-tail p-value

    p_tost = max(p1, p2)

    # 90% CI
    t_crit = float(stats.t.ppf(0.95, df=n - 1))
    ci90_lower = mean_diff - t_crit * se
    ci90_upper = mean_diff + t_crit * se

    return TOSTResult(
        margin=margin,
        mean_diff=mean_diff,
        se=se,
        t1=t1,
        t2=t2,
        p1=p1,
        p2=p2,
        p_tost=p_tost,
        is_equivalent=p_tost < 0.05,
        ci90_lower=ci90_lower,
        ci90_upper=ci90_upper,
    )


# ──────────────────────────────────────────────────────────────────
# NPDE (Comets 2008)
# ──────────────────────────────────────────────────────────────────

def compute_npde(
    observed: NDArray[np.float64],
    simulated_matrix: NDArray[np.float64],
) -> NPDEResult:
    """
    Normalized Prediction Distribution Errors.

    For each observation y_obs:
        pde = (0.5 + sum(y_sim <= y_obs)) / (K + 1)
        npde = Phi^-1(pde)

    Args:
        observed:          Observed concentrations (N,)
        simulated_matrix:  Simulated concentrations (N, K)
                           N observations x K simulations

    Returns:
        NPDEResult with NPDE values and statistical tests
    """
    n = len(observed)
    k = simulated_matrix.shape[1]
    npde_values = np.zeros(n)

    for i in range(n):
        # Empirical CDF
        count_below = np.sum(simulated_matrix[i, :] <= observed[i])
        pde = (0.5 + count_below) / (k + 1)

        # Clamp to avoid infinite values at boundaries
        pde = np.clip(pde, 0.001, 0.999)

        # Inverse normal CDF
        npde_values[i] = float(stats.norm.ppf(pde))

    # Statistical tests
    mean_npde = float(np.mean(npde_values))
    var_npde = float(np.var(npde_values, ddof=1))

    # t-test: H0: mean = 0
    t_stat_mean, p_mean = stats.ttest_1samp(npde_values, 0.0)

    # Variance test (chi-squared): H0: var = 1
    chi2_stat = (n - 1) * var_npde
    p_var = 2.0 * min(
        float(stats.chi2.cdf(chi2_stat, df=n - 1)),
        1.0 - float(stats.chi2.cdf(chi2_stat, df=n - 1)),
    )

    # Shapiro-Wilk normality test
    if n >= 3:
        _, p_normal = stats.shapiro(npde_values)
    else:
        p_normal = 1.0

    is_adequate = (
        float(p_mean) > 0.05
        and p_var > 0.05
        and float(p_normal) > 0.05
    )

    return NPDEResult(
        npde_values=npde_values,
        mean=mean_npde,
        variance=var_npde,
        mean_test_p=float(p_mean),
        variance_test_p=p_var,
        normality_test_p=float(p_normal),
        is_adequate=is_adequate,
    )


# ──────────────────────────────────────────────────────────────────
# Coverage probability
# ──────────────────────────────────────────────────────────────────

def coverage_probability(
    true_values: NDArray[np.float64],
    ci_lower: NDArray[np.float64],
    ci_upper: NDArray[np.float64],
) -> float:
    """
    Compute coverage probability: fraction of true values within CI.

    Target: ~95% for 95% credible intervals.

    Args:
        true_values: True parameter values
        ci_lower:    Lower bounds of credible intervals
        ci_upper:    Upper bounds of credible intervals

    Returns:
        Coverage fraction (0-1)
    """
    covered = (true_values >= ci_lower) & (true_values <= ci_upper)
    return float(np.mean(covered))
