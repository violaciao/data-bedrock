"""High-level Experiment class that wraps stats, sample size, and sequential testing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .sample_size import SampleSizeResult, required_sample_size
from .stats import StatResult, bootstrap_ci, cuped, mann_whitney_u_test, welch_t_test, wilson_ci, z_test

logger = logging.getLogger(__name__)

MetricType = Literal["binary", "continuous"]
TestMethod = Literal["auto", "z_test", "t_test", "mann_whitney", "bootstrap", "cuped"]


@dataclass
class Experiment:
    """Manages a single A/B experiment lifecycle.

    Handles metric type detection, test selection, and result reporting.

    Args:
        name: Human-readable experiment name.
        metric_type: ``"binary"`` for conversion rates, ``"continuous"`` for
            means (revenue, session duration, etc.).
        alpha: Significance level (default 0.05).
        power: Target statistical power for sample size planning (default 0.80).

    Example::

        exp = Experiment("checkout_cta_test", metric_type="binary")
        result = exp.analyze(
            control=df[df.variant == "control"]["converted"].values,
            treatment=df[df.variant == "treatment"]["converted"].values,
        )
        print(result)
    """

    name: str
    metric_type: MetricType = "binary"
    alpha: float = 0.05
    power: float = 0.80

    def plan(
        self,
        baseline_rate: float,
        mde_relative: float | None = None,
        mde_absolute: float | None = None,
        daily_traffic_per_variant: int | None = None,
    ) -> SampleSizeResult:
        """Compute required sample size before running the experiment.

        Args:
            baseline_rate: Current metric value (conversion rate for binary,
                normalised rate for continuous).
            mde_relative: Relative MDE (e.g. 0.10 for a 10% lift).
            mde_absolute: Absolute MDE (e.g. 0.02 for +2pp).
            daily_traffic_per_variant: For experiment duration estimation.

        Returns:
            :class:`~sample_size.SampleSizeResult` with sample size and duration.
        """
        result = required_sample_size(
            baseline_rate=baseline_rate,
            mde_relative=mde_relative,
            mde_absolute=mde_absolute,
            alpha=self.alpha,
            power=self.power,
            daily_traffic_per_variant=daily_traffic_per_variant,
        )
        logger.info("[%s] %s", self.name, result)
        return result

    def analyze(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        method: TestMethod = "auto",
        control_pre: np.ndarray | None = None,
        treatment_pre: np.ndarray | None = None,
    ) -> StatResult:
        """Run the appropriate statistical test and return a result.

        Method selection when ``method="auto"``:
        - ``binary`` metric → z-test
        - ``continuous`` metric with pre-experiment data → CUPED
        - ``continuous`` with skewed data (|skewness| > 2) → Mann-Whitney U
        - otherwise → Welch t-test

        Args:
            control: 1-D array of metric values for control group.
            treatment: 1-D array of metric values for treatment group.
            method: Statistical test to use (default ``"auto"``).
            control_pre: Pre-experiment metric for CUPED (control group).
            treatment_pre: Pre-experiment metric for CUPED (treatment group).

        Returns:
            :class:`~stats.StatResult` with test outcome.
        """
        control = np.asarray(control, dtype=float)
        treatment = np.asarray(treatment, dtype=float)

        resolved_method = self._resolve_method(method, control, treatment, control_pre, treatment_pre)
        logger.info("[%s] using %s (n_c=%d, n_t=%d)", self.name, resolved_method, len(control), len(treatment))

        if resolved_method == "z_test":
            n_c, n_t = len(control), len(treatment)
            conv_c = int(control.sum())
            conv_t = int(treatment.sum())
            return z_test(conv_c, n_c, conv_t, n_t, alpha=self.alpha)

        if resolved_method == "cuped":
            assert control_pre is not None and treatment_pre is not None
            return cuped(control, treatment, control_pre, treatment_pre, alpha=self.alpha)

        if resolved_method == "mann_whitney":
            return mann_whitney_u_test(control, treatment, alpha=self.alpha)

        # Default: Welch t-test
        return welch_t_test(control, treatment, alpha=self.alpha)

    def _resolve_method(
        self,
        method: TestMethod,
        control: np.ndarray,
        treatment: np.ndarray,
        control_pre: np.ndarray | None,
        treatment_pre: np.ndarray | None,
    ) -> str:
        if method != "auto":
            return method
        if self.metric_type == "binary":
            return "z_test"
        if control_pre is not None and treatment_pre is not None:
            return "cuped"
        from scipy.stats import skew
        pooled_skew = abs(float(skew(np.concatenate([control, treatment]))))
        if pooled_skew > 2:
            logger.debug("Skewness=%.2f — using Mann-Whitney", pooled_skew)
            return "mann_whitney"
        return "t_test"

    def summary(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        result: StatResult,
    ) -> pd.DataFrame:
        """Return a tidy summary DataFrame for reporting.

        Args:
            control: Control group values.
            treatment: Treatment group values.
            result: The :class:`~stats.StatResult` from :meth:`analyze`.

        Returns:
            Single-row DataFrame with key metrics.
        """
        control = np.asarray(control, dtype=float)
        treatment = np.asarray(treatment, dtype=float)

        return pd.DataFrame(
            [
                {
                    "experiment": self.name,
                    "method": result.method,
                    "n_control": len(control),
                    "n_treatment": len(treatment),
                    "mean_control": control.mean(),
                    "mean_treatment": treatment.mean(),
                    "effect_size": result.effect_size,
                    "ci_lower": result.ci_lower,
                    "ci_upper": result.ci_upper,
                    "p_value": result.p_value,
                    "alpha": result.alpha,
                    "significant": result.significant,
                }
            ]
        )
