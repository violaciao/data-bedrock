"""Difference-in-Differences (DiD) estimator.

DiD identifies a causal effect by comparing the change over time in a treated
group against the change over time in an untreated control group:

    ATT = (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre)

The key assumption is **parallel trends**: in the absence of treatment, both
groups would have followed the same time trend.

This module provides:
- Classic 2×2 DiD (one pre period, one post period)
- Two-Way Fixed Effects (TWFE) regression DiD for panel data
- Parallel trends pre-test (event study up to treatment date)

Design decisions:
- We implement DiD via OLS (statsmodels) so you get standard errors and
  confidence intervals, not just a point estimate.
- TWFE via OLS handles multiple units and multiple time periods. For
  staggered adoption, see the decision log — TWFE is biased there and
  a heterogeneity-robust estimator (Callaway-Sant'Anna) is preferred.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiDResult:
    """Result from a Difference-in-Differences estimation.

    Attributes:
        att: Average Treatment effect on the Treated (point estimate).
        std_err: Standard error of the ATT estimate.
        ci_lower: Lower bound of the (1-alpha) confidence interval.
        ci_upper: Upper bound of the (1-alpha) confidence interval.
        p_value: Two-sided p-value for H0: ATT = 0.
        alpha: Significance level used.
        significant: True if p_value < alpha.
        pre_mean_treated: Pre-period mean for the treated group.
        pre_mean_control: Pre-period mean for the control group.
        post_mean_treated: Post-period mean for the treated group.
        post_mean_control: Post-period mean for the control group.
        method: Estimation method used.
    """

    att: float
    std_err: float
    ci_lower: float
    ci_upper: float
    p_value: float
    alpha: float
    significant: bool
    pre_mean_treated: float
    pre_mean_control: float
    post_mean_treated: float
    post_mean_control: float
    method: str

    def __str__(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"DiD ATT = {self.att:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}] "
            f"(p={self.p_value:.4f}, {sig} at α={self.alpha})\n"
            f"  Pre:  treated={self.pre_mean_treated:.4f}  control={self.pre_mean_control:.4f}\n"
            f"  Post: treated={self.post_mean_treated:.4f}  control={self.post_mean_control:.4f}"
        )


class DifferenceInDifferences:
    """Difference-in-Differences causal estimator.

    Args:
        alpha: Significance level for confidence intervals (default 0.05).

    Example — classic 2×2 DiD::

        did = DifferenceInDifferences()
        result = did.estimate_2x2(
            pre_treated=[10.1, 9.8, 10.3],
            post_treated=[12.5, 12.1, 12.8],
            pre_control=[8.9, 9.2, 8.7],
            post_control=[9.1, 9.4, 9.0],
        )
        print(result)

    Example — panel DiD via TWFE::

        result = did.estimate_panel(
            df=panel_df,
            unit_col="store_id",
            time_col="week",
            outcome_col="sales",
            treated_col="is_treated",
            treatment_time=pd.Timestamp("2024-06-01"),
        )
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def estimate_2x2(
        self,
        pre_treated: np.ndarray,
        post_treated: np.ndarray,
        pre_control: np.ndarray,
        post_control: np.ndarray,
    ) -> DiDResult:
        """Classic 2×2 DiD with standard errors via OLS.

        Equivalent to regressing outcome on (treated × post) with
        unit and time fixed effects in the 2-group, 2-period case.

        Args:
            pre_treated: Outcome values for the treated group before treatment.
            post_treated: Outcome values for the treated group after treatment.
            pre_control: Outcome values for the control group before treatment.
            post_control: Outcome values for the control group after treatment.

        Returns:
            :class:`DiDResult` with ATT estimate and inference.
        """
        pre_treated = np.asarray(pre_treated, dtype=float)
        post_treated = np.asarray(post_treated, dtype=float)
        pre_control = np.asarray(pre_control, dtype=float)
        post_control = np.asarray(post_control, dtype=float)

        # Stack into a DataFrame for OLS
        n = len(pre_treated) + len(post_treated) + len(pre_control) + len(post_control)
        treated_flag = np.concatenate([
            np.ones(len(pre_treated) + len(post_treated)),
            np.zeros(len(pre_control) + len(post_control)),
        ])
        post_flag = np.concatenate([
            np.zeros(len(pre_treated)), np.ones(len(post_treated)),
            np.zeros(len(pre_control)), np.ones(len(post_control)),
        ])
        outcome = np.concatenate([pre_treated, post_treated, pre_control, post_control])
        interaction = treated_flag * post_flag

        # OLS: Y = β0 + β1·treated + β2·post + β3·(treated×post) + ε
        # β3 is the DiD estimator (ATT)
        X = np.column_stack([np.ones(n), treated_flag, post_flag, interaction])
        result = _ols_with_inference(X, outcome, self.alpha)
        att = result["coef"][3]
        se = result["se"][3]
        t_crit = scipy_stats.t.ppf(1 - self.alpha / 2, df=result["df"])
        p_value = float(2 * scipy_stats.t.sf(abs(att / se), df=result["df"]))

        return DiDResult(
            att=att,
            std_err=se,
            ci_lower=att - t_crit * se,
            ci_upper=att + t_crit * se,
            p_value=p_value,
            alpha=self.alpha,
            significant=p_value < self.alpha,
            pre_mean_treated=float(pre_treated.mean()),
            pre_mean_control=float(pre_control.mean()),
            post_mean_treated=float(post_treated.mean()),
            post_mean_control=float(post_control.mean()),
            method="did_2x2_ols",
        )

    def estimate_panel(
        self,
        df: pd.DataFrame,
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treated_col: str,
        treatment_time: object,
    ) -> DiDResult:
        """Two-Way Fixed Effects (TWFE) DiD for panel data.

        Regresses the outcome on a post×treated interaction with unit and
        time fixed effects absorbed via demeaning.

        **Warning:** TWFE is biased with staggered treatment adoption (different
        units treated at different times). In that case, use a
        heterogeneity-robust estimator (Callaway-Sant'Anna 2021).

        Args:
            df: Panel DataFrame with one row per (unit, time) observation.
            unit_col: Column identifying units (e.g. ``"store_id"``).
            time_col: Column with the time period.
            outcome_col: Column with the outcome variable.
            treated_col: Boolean column indicating treated units.
            treatment_time: The time value at which treatment begins.

        Returns:
            :class:`DiDResult` with TWFE ATT estimate.
        """
        df = df.copy()
        df["_post"] = (df[time_col] >= treatment_time).astype(float)
        df["_did"] = df[treated_col].astype(float) * df["_post"]

        # Demean to absorb unit and time fixed effects
        df["_y"] = df[outcome_col].astype(float)
        for col in ["_y", "_did", "_post"]:
            df[col] -= df.groupby(unit_col)[col].transform("mean")
            df[col] -= df.groupby(time_col)[col].transform("mean")
            df[col] += df[col].mean() + df[outcome_col].mean()

        X = np.column_stack([np.ones(len(df)), df["_post"].values, df["_did"].values])
        result = _ols_with_inference(X, df["_y"].values, self.alpha)
        att = result["coef"][2]
        se = result["se"][2]
        t_crit = scipy_stats.t.ppf(1 - self.alpha / 2, df=result["df"])
        p_value = float(2 * scipy_stats.t.sf(abs(att / se), df=result["df"]))

        treated_mask = df[treated_col].astype(bool)
        pre_mask = df[time_col] < treatment_time
        return DiDResult(
            att=att,
            std_err=se,
            ci_lower=att - t_crit * se,
            ci_upper=att + t_crit * se,
            p_value=p_value,
            alpha=self.alpha,
            significant=p_value < self.alpha,
            pre_mean_treated=float(df.loc[treated_mask & pre_mask, outcome_col].mean()),
            pre_mean_control=float(df.loc[~treated_mask & pre_mask, outcome_col].mean()),
            post_mean_treated=float(df.loc[treated_mask & ~pre_mask, outcome_col].mean()),
            post_mean_control=float(df.loc[~treated_mask & ~pre_mask, outcome_col].mean()),
            method="twfe_panel",
        )

    def event_study(
        self,
        df: pd.DataFrame,
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treated_col: str,
        treatment_time: object,
        n_pre: int = 4,
        n_post: int = 4,
    ) -> pd.DataFrame:
        """Event study: per-period ATT estimates relative to treatment.

        Estimates the treatment effect at each period relative to treatment
        (t = -n_pre, …, -1, 0, 1, …, n_post). The pre-period estimates
        (t < 0) should be near zero — a visual parallel trends test.

        Args:
            df: Panel DataFrame.
            unit_col: Unit identifier column.
            time_col: Time column (must be sortable).
            outcome_col: Outcome column.
            treated_col: Boolean treated indicator.
            treatment_time: Treatment start time.
            n_pre: Number of pre-treatment periods to include.
            n_post: Number of post-treatment periods to include.

        Returns:
            DataFrame with columns: relative_period, att, ci_lower, ci_upper.
        """
        df = df.copy()
        all_times = sorted(df[time_col].unique())
        t0_idx = all_times.index(treatment_time)
        window = all_times[max(0, t0_idx - n_pre): t0_idx + n_post + 1]

        rows = []
        for period in window:
            rel = all_times.index(period) - t0_idx
            sub = df[df[time_col].isin([all_times[max(0, t0_idx - 1)], period])].copy()
            if len(sub[time_col].unique()) < 2:
                continue
            try:
                res = self.estimate_panel(sub, unit_col, time_col, outcome_col, treated_col, period)
                rows.append({"relative_period": rel, "att": res.att, "ci_lower": res.ci_lower, "ci_upper": res.ci_upper})
            except Exception:
                pass

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ols_with_inference(X: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    """OLS via normal equations with HC1 heteroskedasticity-robust standard errors."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    residuals = y - X @ coef

    # HC1 sandwich covariance: (XtX)^-1 · X' diag(e²·n/(n-k)) X · (XtX)^-1
    scale = n / (n - k)
    meat = (X * (residuals ** 2 * scale)[:, None]).T @ X
    vcov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(vcov))
    df = n - k
    return {"coef": coef, "se": se, "vcov": vcov, "df": df, "residuals": residuals}
