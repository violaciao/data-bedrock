"""Synthetic Control Method.

The Synthetic Control Method constructs a weighted combination of control
units that closely matches the treated unit in the pre-treatment period.
Post-treatment, the gap between the treated unit and its synthetic counterpart
estimates the causal effect.

Key advantages over DiD:
- No parallel trends assumption required
- Works with a single treated unit (e.g. one country, one store)
- Transparent: the weights show which controls contributed

Key limitations:
- Requires multiple pre-treatment periods for a good fit
- Inference is done via permutation (placebo) tests, not standard errors
- Extrapolation outside the convex hull of donor units is not justified

Reference:
    Abadie, Diamond, Hainmueller (2010). "Synthetic Control Methods for
    Comparative Case Studies". JASA 105(490):493–505.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class SyntheticControlResult:
    """Result from a Synthetic Control estimation.

    Attributes:
        weights: Series mapping donor unit → weight (sum to 1).
        pre_rmspe: Root Mean Squared Prediction Error in the pre period.
        att_series: Series of (post_period_index → ATT) estimates.
        att_mean: Mean ATT over the post-treatment periods.
        synthetic_outcome: Full time series of the synthetic control outcome.
        treated_outcome: Full time series of the treated unit outcome.
    """

    weights: pd.Series
    pre_rmspe: float
    att_series: pd.Series
    att_mean: float
    synthetic_outcome: pd.Series
    treated_outcome: pd.Series

    def __str__(self) -> str:
        top_donors = self.weights[self.weights > 0.01].sort_values(ascending=False).head(5)
        return (
            f"Synthetic Control ATT (mean post) = {self.att_mean:.4f}\n"
            f"  Pre-period RMSPE = {self.pre_rmspe:.4f}\n"
            f"  Top donors: {dict(top_donors.round(3))}"
        )


class SyntheticControl:
    """Synthetic Control Method estimator.

    Args:
        alpha: Not used for inference (inference is via placebo). Kept for
            API consistency with other estimators.

    Example::

        sc = SyntheticControl()
        result = sc.fit(
            df=panel_df,           # wide format: rows=time, cols=units
            treated_unit="store_A",
            treatment_time=12,     # time index (row index) of treatment
        )
        print(result)
        # result.weights shows which donor units were used
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def fit(
        self,
        df: pd.DataFrame,
        treated_unit: str,
        treatment_time: int | str,
    ) -> SyntheticControlResult:
        """Estimate the synthetic control and treatment effect.

        Args:
            df: Wide-format DataFrame where the index is time and each column
                is a unit. One column is the treated unit; the rest are donors.
            treated_unit: Column name of the treated unit.
            treatment_time: Index value of the first post-treatment period.

        Returns:
            :class:`SyntheticControlResult`.
        """
        df = df.copy()
        donor_units = [c for c in df.columns if c != treated_unit]

        pre_mask = df.index < treatment_time
        post_mask = df.index >= treatment_time

        Y_treated = df[treated_unit].values
        Y_donors = df[donor_units].values  # shape (T, n_donors)

        Y_pre_treated = Y_treated[pre_mask]
        Y_pre_donors = Y_donors[pre_mask]

        # Find weights W ≥ 0 that sum to 1 minimising ||Y_pre_treated - Y_pre_donors @ W||²
        n_donors = len(donor_units)
        w0 = np.ones(n_donors) / n_donors

        def objective(w: np.ndarray) -> float:
            return float(np.sum((Y_pre_treated - Y_pre_donors @ w) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0, 1.0)] * n_donors
        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)

        weights = res.x
        weights = np.clip(weights, 0, 1)
        weights /= weights.sum()  # re-normalise after clipping

        synthetic = Y_donors @ weights
        pre_rmspe = float(np.sqrt(np.mean((Y_pre_treated - synthetic[pre_mask]) ** 2)))

        # ATT = treated - synthetic, post-treatment only
        att_vals = Y_treated[post_mask] - synthetic[post_mask]
        att_series = pd.Series(att_vals, index=df.index[post_mask], name="att")

        logger.info(
            "Synthetic Control: pre_rmspe=%.4f  mean_att=%.4f  top_donor=%s(%.2f)",
            pre_rmspe,
            float(att_vals.mean()),
            donor_units[int(np.argmax(weights))],
            weights.max(),
        )

        return SyntheticControlResult(
            weights=pd.Series(weights, index=donor_units, name="weight"),
            pre_rmspe=pre_rmspe,
            att_series=att_series,
            att_mean=float(att_vals.mean()),
            synthetic_outcome=pd.Series(synthetic, index=df.index, name="synthetic"),
            treated_outcome=pd.Series(Y_treated, index=df.index, name=treated_unit),
        )

    def placebo_test(
        self,
        df: pd.DataFrame,
        treated_unit: str,
        treatment_time: int | str,
    ) -> pd.DataFrame:
        """Run placebo tests on all donor units to assess inference.

        Applies the synthetic control to each donor unit as if it were
        treated. The ratio of post-RMSPE to pre-RMSPE for the treated unit
        vs donors gives an informal p-value.

        Args:
            df: Same DataFrame passed to :meth:`fit`.
            treated_unit: The true treated unit.
            treatment_time: Treatment start index.

        Returns:
            DataFrame with columns: unit, pre_rmspe, post_rmspe, rmspe_ratio.
            Sorted by rmspe_ratio descending.
        """
        all_units = list(df.columns)
        rows = []
        for unit in all_units:
            donors = [c for c in df.columns if c != unit]
            if len(donors) < 2:
                continue
            try:
                result = self.fit(df[[unit] + donors], treated_unit=unit, treatment_time=treatment_time)
                post_mask = df.index >= treatment_time
                post_rmspe = float(np.sqrt(np.mean(result.att_series.values ** 2)))
                rows.append({
                    "unit": unit,
                    "pre_rmspe": result.pre_rmspe,
                    "post_rmspe": post_rmspe,
                    "rmspe_ratio": post_rmspe / (result.pre_rmspe + 1e-10),
                    "is_treated": unit == treated_unit,
                })
            except Exception as e:
                logger.debug("Placebo failed for unit %s: %s", unit, e)

        return pd.DataFrame(rows).sort_values("rmspe_ratio", ascending=False)
