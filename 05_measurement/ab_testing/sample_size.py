"""Sample size calculator for A/B tests.

Design decisions:
- We support both relative and absolute MDE so callers can express "I want to
  detect a 10% lift" or "I want to detect a +2pp change" equally naturally.
- Experiment duration estimation requires a daily traffic number, which is a
  business input — we keep it as an explicit parameter rather than hiding it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats as scipy_stats


@dataclass(frozen=True)
class SampleSizeResult:
    """Result from :func:`required_sample_size`.

    Attributes:
        n_per_variant: Minimum observations needed per variant.
        n_total: Total observations across all variants.
        baseline_rate: Baseline conversion rate used in calculation.
        mde_absolute: Absolute minimum detectable effect.
        alpha: Type I error rate.
        power: Statistical power (1 - beta).
        estimated_days: Estimated experiment duration in days, or None if
            ``daily_traffic_per_variant`` was not provided.
    """

    n_per_variant: int
    n_total: int
    baseline_rate: float
    mde_absolute: float
    alpha: float
    power: float
    estimated_days: int | None

    def __str__(self) -> str:
        duration = f"  estimated duration: {self.estimated_days} days\n" if self.estimated_days else ""
        return (
            f"Sample size required\n"
            f"  per variant:   {self.n_per_variant:,}\n"
            f"  total:         {self.n_total:,}\n"
            f"  baseline rate: {self.baseline_rate:.2%}\n"
            f"  MDE:           {self.mde_absolute:+.4f} ({self.mde_absolute / self.baseline_rate:+.1%} relative)\n"
            f"  α={self.alpha}  power={self.power:.0%}\n"
            f"{duration}"
        )


def required_sample_size(
    baseline_rate: float,
    mde_relative: float | None = None,
    mde_absolute: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    n_variants: int = 2,
    daily_traffic_per_variant: int | None = None,
) -> SampleSizeResult:
    """Compute the minimum sample size for a binary metric A/B test.

    Exactly one of ``mde_relative`` or ``mde_absolute`` must be provided.

    The formula uses the standard two-proportion z-test power analysis:

        n = (z_alpha/2 + z_beta)^2 * (p1*(1-p1) + p2*(1-p2)) / (p1 - p2)^2

    Args:
        baseline_rate: Current conversion rate (0–1). E.g. ``0.05`` for 5%.
        mde_relative: Minimum detectable effect as a relative lift. E.g. ``0.10``
            means detect a 10% relative improvement (5% → 5.5%).
        mde_absolute: Minimum detectable effect as an absolute difference.
            E.g. ``0.02`` means detect a +2pp change (5% → 7%).
        alpha: Type I error rate (default 0.05, two-tailed).
        power: Desired statistical power (default 0.80).
        n_variants: Number of variants including control (default 2).
            Used only to compute ``n_total``.
        daily_traffic_per_variant: Average daily observations routed to each
            variant. If provided, ``estimated_days`` is returned.

    Returns:
        :class:`SampleSizeResult` with sample sizes and optional duration.

    Raises:
        ValueError: If neither or both of MDE parameters are provided, or if
            inputs are out of range.
    """
    if (mde_relative is None) == (mde_absolute is None):
        raise ValueError("Provide exactly one of mde_relative or mde_absolute.")
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be in (0, 1).")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1).")

    if mde_relative is not None:
        delta = baseline_rate * mde_relative
    else:
        delta = mde_absolute  # type: ignore[assignment]

    treatment_rate = baseline_rate + delta
    if not 0 < treatment_rate < 1:
        raise ValueError(
            f"Implied treatment rate {treatment_rate:.4f} is outside (0, 1). "
            "Adjust baseline_rate or MDE."
        )

    z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
    z_beta = scipy_stats.norm.ppf(power)

    p1, p2 = baseline_rate, treatment_rate
    numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denominator = (p2 - p1) ** 2
    n = math.ceil(numerator / denominator)

    estimated_days: int | None = None
    if daily_traffic_per_variant is not None and daily_traffic_per_variant > 0:
        estimated_days = math.ceil(n / daily_traffic_per_variant)

    return SampleSizeResult(
        n_per_variant=n,
        n_total=n * n_variants,
        baseline_rate=baseline_rate,
        mde_absolute=delta,
        alpha=alpha,
        power=power,
        estimated_days=estimated_days,
    )
