"""Sequential (always-valid) hypothesis testing via mSPRT.

Standard fixed-horizon tests break when you peek at results early: the
false-positive rate inflates because you implicitly do multiple comparisons.
The mixture Sequential Probability Ratio Test (mSPRT) gives you valid
p-values at every observation — you can stop as soon as p < alpha.

Design decisions:
- We implement the normal-mixture mSPRT (Johari et al. 2017) which works for
  continuous metrics with known or estimated variance.
- The mixing variance `tau_sq` controls how sensitive the test is. A larger
  tau_sq detects larger effects sooner; a smaller one has more power for tiny
  effects. Default of 1.0 works well for standardised metrics.

Reference:
    Johari, Pekelis, Walsh (2017) "Always Valid Inference: Bringing
    Sequential Analysis to A/B Testing". arXiv:1512.04922.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SequentialResult:
    """Snapshot result from :class:`SequentialTest`.

    Attributes:
        n_control: Current control sample size.
        n_treatment: Current treatment sample size.
        effect_size: Current point estimate (mean_t - mean_c).
        p_value: Always-valid p-value at this point.
        alpha: Significance level.
        significant: True if p_value < alpha.
        log_likelihood_ratio: Raw log of the likelihood ratio (llr ≥ log(1/alpha) ⟹ reject).
    """

    n_control: int
    n_treatment: int
    effect_size: float
    p_value: float
    alpha: float
    significant: bool
    log_likelihood_ratio: float


@dataclass
class SequentialTest:
    """Stateful sequential test that can be updated observation-by-observation.

    Uses the normal-mixture mSPRT. Each call to :meth:`update` adds new
    observations and returns the current :class:`SequentialResult`.

    Example::

        test = SequentialTest(alpha=0.05)
        for batch in data_stream:
            result = test.update(batch["control"], batch["treatment"])
            if result.significant:
                print("Stop the experiment:", result)
                break

    Args:
        alpha: Significance level (default 0.05).
        tau_sq: Mixing variance parameter (default 1.0). Controls sensitivity.
            Tune to your expected effect size range.
    """

    alpha: float = 0.05
    tau_sq: float = 1.0

    # Running statistics (updated incrementally)
    _n_c: int = field(default=0, init=False, repr=False)
    _n_t: int = field(default=0, init=False, repr=False)
    _sum_c: float = field(default=0.0, init=False, repr=False)
    _sum_t: float = field(default=0.0, init=False, repr=False)
    _sum_sq_c: float = field(default=0.0, init=False, repr=False)
    _sum_sq_t: float = field(default=0.0, init=False, repr=False)

    def update(
        self,
        control_obs: np.ndarray,
        treatment_obs: np.ndarray,
    ) -> SequentialResult:
        """Add new observations and return the current test result.

        Args:
            control_obs: New control observations (1-D array).
            treatment_obs: New treatment observations (1-D array).

        Returns:
            :class:`SequentialResult` reflecting all observations so far.
        """
        control_obs = np.asarray(control_obs, dtype=float)
        treatment_obs = np.asarray(treatment_obs, dtype=float)

        self._n_c += len(control_obs)
        self._n_t += len(treatment_obs)
        self._sum_c += float(control_obs.sum())
        self._sum_t += float(treatment_obs.sum())
        self._sum_sq_c += float((control_obs ** 2).sum())
        self._sum_sq_t += float((treatment_obs ** 2).sum())

        return self._compute_result()

    def _compute_result(self) -> SequentialResult:
        """Compute the mSPRT statistic from running totals."""
        n_c, n_t = self._n_c, self._n_t
        if n_c < 2 or n_t < 2:
            return SequentialResult(
                n_control=n_c,
                n_treatment=n_t,
                effect_size=0.0,
                p_value=1.0,
                alpha=self.alpha,
                significant=False,
                log_likelihood_ratio=0.0,
            )

        mean_c = self._sum_c / n_c
        mean_t = self._sum_t / n_t
        effect = mean_t - mean_c

        # Pooled variance estimate
        var_c = (self._sum_sq_c - n_c * mean_c ** 2) / (n_c - 1)
        var_t = (self._sum_sq_t - n_t * mean_t ** 2) / (n_t - 1)
        sigma_sq = (var_c + var_t) / 2  # pooled
        sigma_sq = max(sigma_sq, 1e-10)  # guard against zero variance

        # Variance of the treatment effect estimator (δ̂ = ȳ_T − ȳ_C)
        sigma_delta_sq = sigma_sq * (1.0 / n_c + 1.0 / n_t)
        sigma_delta_sq = max(sigma_delta_sq, 1e-10)

        # mSPRT log-likelihood ratio (normal mixture prior on δ ~ N(0, τ²))
        # Derived from: LLR = ∫ f(δ̂; δ, σ²_δ)/f(δ̂; 0, σ²_δ) · dN(δ; 0, τ²)
        # = 0.5·log(σ²_δ/(σ²_δ+τ²)) + τ²·δ̂²/(2σ²_δ(σ²_δ+τ²))
        tau_sq = self.tau_sq
        denom = sigma_delta_sq + tau_sq
        llr = 0.5 * np.log(sigma_delta_sq / denom) + (tau_sq * effect ** 2) / (2 * sigma_delta_sq * denom)

        # Always-valid p-value: p = min(1, exp(-llr))
        p_value = float(min(1.0, np.exp(-llr)))

        return SequentialResult(
            n_control=n_c,
            n_treatment=n_t,
            effect_size=effect,
            p_value=p_value,
            alpha=self.alpha,
            significant=p_value < self.alpha,
            log_likelihood_ratio=float(llr),
        )

    def reset(self) -> None:
        """Reset all running statistics (start a new experiment)."""
        self._n_c = 0
        self._n_t = 0
        self._sum_c = 0.0
        self._sum_t = 0.0
        self._sum_sq_c = 0.0
        self._sum_sq_t = 0.0
