"""Core statistical functions for A/B testing.

All public functions return a StatResult namedtuple so callers always get
effect size, CI, and p-value together — never a bare float.

Design decisions:
- We wrap scipy rather than calling it directly so that every call site makes
  assumptions explicit (two-tailed vs one-tailed, equal vs unequal variance).
- CUPED is implemented as a covariate adjustment (pre-experiment metric) to
  reduce variance and improve power without touching the sample size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StatResult:
    """Result returned by every test function.

    Attributes:
        statistic: Test statistic value (z, t, U, etc.).
        p_value: Two-tailed p-value.
        effect_size: Point estimate of the treatment effect (mean_b - mean_a
            for continuous metrics, rate_b - rate_a for proportions).
        ci_lower: Lower bound of the (1-alpha) confidence interval on effect_size.
        ci_upper: Upper bound of the (1-alpha) confidence interval on effect_size.
        alpha: Significance level used.
        significant: True if p_value < alpha.
        method: Name of the statistical test.
    """

    statistic: float
    p_value: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    alpha: float
    significant: bool
    method: str

    def __str__(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.method}: effect={self.effect_size:.4f} "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"p={self.p_value:.4f} ({sig} at α={self.alpha})"
        )


# ---------------------------------------------------------------------------
# Two-proportion z-test
# ---------------------------------------------------------------------------


def z_test(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> StatResult:
    """Two-proportion z-test for conversion rates.

    Use this when your metric is a binary outcome (converted / not converted).
    Assumes large enough samples that the normal approximation holds (n*p > 5
    and n*(1-p) > 5 for both groups).

    Args:
        control_conversions: Number of conversions in control.
        control_n: Total observations in control.
        treatment_conversions: Number of conversions in treatment.
        treatment_n: Total observations in treatment.
        alpha: Significance level (default 0.05).

    Returns:
        StatResult with z-statistic, p-value, and CI on rate difference.
    """
    if control_n == 0 or treatment_n == 0:
        raise ValueError("control_n and treatment_n must be > 0.")
    p_c = control_conversions / control_n
    p_t = treatment_conversions / treatment_n
    p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)

    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
    if se_pool == 0:
        raise ValueError("Pooled standard error is zero — check your inputs.")

    z = (p_t - p_c) / se_pool
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

    # CI on the difference using unpooled SE (for estimation, not testing)
    se_diff = np.sqrt(p_c * (1 - p_c) / control_n + p_t * (1 - p_t) / treatment_n)
    z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
    effect = p_t - p_c

    return StatResult(
        statistic=z,
        p_value=p_value,
        effect_size=effect,
        ci_lower=effect - z_crit * se_diff,
        ci_upper=effect + z_crit * se_diff,
        alpha=alpha,
        significant=p_value < alpha,
        method="two_proportion_z_test",
    )


# ---------------------------------------------------------------------------
# Welch's t-test
# ---------------------------------------------------------------------------


def welch_t_test(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> StatResult:
    """Welch's two-sample t-test for continuous metrics.

    Welch's variant does not assume equal variances — this is almost always
    the right choice over Student's t-test in practice.

    Args:
        control: 1-D array of metric values for control group.
        treatment: 1-D array of metric values for treatment group.
        alpha: Significance level (default 0.05).

    Returns:
        StatResult with t-statistic, p-value, and CI on mean difference.
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    t_stat, p_value = scipy_stats.ttest_ind(treatment, control, equal_var=False)
    effect = treatment.mean() - control.mean()

    # CI using Welch–Satterthwaite degrees of freedom
    n_c, n_t = len(control), len(treatment)
    s_c, s_t = control.var(ddof=1), treatment.var(ddof=1)
    se = np.sqrt(s_c / n_c + s_t / n_t)
    df_num = (s_c / n_c + s_t / n_t) ** 2
    df_den = (s_c / n_c) ** 2 / (n_c - 1) + (s_t / n_t) ** 2 / (n_t - 1)
    df = df_num / df_den if df_den > 0 else min(n_c, n_t) - 1
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=df)

    return StatResult(
        statistic=t_stat,
        p_value=p_value,
        effect_size=effect,
        ci_lower=effect - t_crit * se,
        ci_upper=effect + t_crit * se,
        alpha=alpha,
        significant=p_value < alpha,
        method="welch_t_test",
    )


# ---------------------------------------------------------------------------
# Mann-Whitney U test
# ---------------------------------------------------------------------------


def mann_whitney_u_test(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> StatResult:
    """Mann-Whitney U test (non-parametric alternative to t-test).

    Use this when the metric is skewed (e.g., revenue, session duration) and
    you cannot assume normality even with large samples. The effect size here
    is the median difference (not the mean difference).

    The CI is computed via bootstrap since there is no closed-form solution
    for the median difference CI.

    Args:
        control: 1-D array of metric values for control group.
        treatment: 1-D array of metric values for treatment group.
        alpha: Significance level (default 0.05).

    Returns:
        StatResult with U-statistic, p-value, and bootstrapped CI on median diff.
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    u_stat, p_value = scipy_stats.mannwhitneyu(treatment, control, alternative="two-sided")
    effect = float(np.median(treatment) - np.median(control))

    # Bootstrap CI for median difference
    ci = bootstrap_ci(control, treatment, statistic="median", alpha=alpha, n_bootstrap=2000)

    return StatResult(
        statistic=u_stat,
        p_value=p_value,
        effect_size=effect,
        ci_lower=ci[0],
        ci_upper=ci[1],
        alpha=alpha,
        significant=p_value < alpha,
        method="mann_whitney_u",
    )


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    control: np.ndarray,
    treatment: np.ndarray,
    statistic: str = "mean",
    alpha: float = 0.05,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a treatment effect.

    Args:
        control: 1-D array of control group values.
        treatment: 1-D array of treatment group values.
        statistic: ``"mean"`` or ``"median"`` (default ``"mean"``).
        alpha: Significance level — returns (alpha/2, 1-alpha/2) quantile interval.
        n_bootstrap: Number of bootstrap resamples (default 5000).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if statistic not in ("mean", "median"):
        raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}")
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    rng = np.random.default_rng(seed)
    stat_fn = np.mean if statistic == "mean" else np.median

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        c_boot = rng.choice(control, size=len(control), replace=True)
        t_boot = rng.choice(treatment, size=len(treatment), replace=True)
        diffs[i] = stat_fn(t_boot) - stat_fn(c_boot)

    lower = float(np.percentile(diffs, 100 * alpha / 2))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return lower, upper


# ---------------------------------------------------------------------------
# Wilson CI for proportions
# ---------------------------------------------------------------------------


def wilson_ci(
    conversions: int,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Prefer Wilson over the normal approximation (Wald) interval because it
    never goes outside [0, 1] and has better coverage near 0 and 1.

    Args:
        conversions: Number of successes.
        n: Total observations.
        alpha: Significance level (default 0.05).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if n == 0:
        raise ValueError("n must be > 0")

    p_hat = conversions / n
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    z2 = z ** 2
    denominator = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n ** 2)) / denominator
    return float(centre - margin), float(centre + margin)


# ---------------------------------------------------------------------------
# CUPED variance reduction
# ---------------------------------------------------------------------------


def cuped(
    control_post: np.ndarray,
    treatment_post: np.ndarray,
    control_pre: np.ndarray,
    treatment_pre: np.ndarray,
    alpha: float = 0.05,
) -> StatResult:
    """CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

    Adjusts the post-experiment metric by removing variance explained by the
    pre-experiment metric (e.g., last week's revenue). This is equivalent to
    ANCOVA and can substantially increase statistical power.

    The adjustment coefficient theta is estimated from the pooled data:
        theta = Cov(Y_post, Y_pre) / Var(Y_pre)
        Y_adjusted = Y_post - theta * (Y_pre - mean(Y_pre))

    Args:
        control_post: Post-experiment metric for control group.
        treatment_post: Post-experiment metric for treatment group.
        control_pre: Pre-experiment metric for the same control users.
        treatment_pre: Pre-experiment metric for the same treatment users.
        alpha: Significance level (default 0.05).

    Returns:
        StatResult from a Welch t-test on the CUPED-adjusted metric.
    """
    control_post = np.asarray(control_post, dtype=float)
    treatment_post = np.asarray(treatment_post, dtype=float)
    control_pre = np.asarray(control_pre, dtype=float)
    treatment_pre = np.asarray(treatment_pre, dtype=float)

    # Pool pre-experiment data to estimate theta
    all_post = np.concatenate([control_post, treatment_post])
    all_pre = np.concatenate([control_pre, treatment_pre])
    pre_mean = all_pre.mean()

    cov_matrix = np.cov(all_post, all_pre)
    theta = cov_matrix[0, 1] / cov_matrix[1, 1]

    logger.debug("CUPED theta=%.4f (variance reduction ≈ %.1f%%)", theta, 100 * theta ** 2 * np.var(all_pre) / np.var(all_post))

    control_adj = control_post - theta * (control_pre - pre_mean)
    treatment_adj = treatment_post - theta * (treatment_pre - pre_mean)

    result = welch_t_test(control_adj, treatment_adj, alpha=alpha)
    return StatResult(
        statistic=result.statistic,
        p_value=result.p_value,
        effect_size=result.effect_size,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        alpha=result.alpha,
        significant=result.significant,
        method="cuped",
    )
