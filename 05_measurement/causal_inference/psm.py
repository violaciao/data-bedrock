"""Propensity Score Matching (PSM).

PSM estimates the causal effect of a binary treatment by matching treated
units to similar control units based on their probability of receiving
treatment (propensity score), then comparing outcomes within matched pairs.

The key assumption is **unconfoundedness** (a.k.a. selection on observables):
conditional on observed covariates, treatment assignment is independent of
the potential outcomes. PSM does not handle unobserved confounders.

This module provides:
- Propensity score estimation via logistic regression
- Nearest-neighbour matching (with optional caliper)
- ATT and ATE estimation from matched samples
- Covariate balance diagnostics (Standardised Mean Difference)

Design decisions:
- We use logistic regression rather than a black-box ML model for the
  propensity score. This makes the model transparent and auditable.
  For complex high-dimensional covariates, you could substitute any
  classifier that outputs probabilities.
- We implement 1:k nearest-neighbour matching without replacement.
  Matching with replacement gives lower bias but higher variance; the
  tradeoff depends on dataset size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchingResult:
    """Result from propensity score matching.

    Attributes:
        att: Average Treatment effect on the Treated.
        ate: Average Treatment Effect (over full matched sample).
        std_err_att: Standard error of ATT.
        ci_lower_att: Lower CI bound for ATT.
        ci_upper_att: Upper CI bound for ATT.
        p_value_att: Two-sided p-value for ATT.
        n_treated: Number of treated units.
        n_matched_control: Number of matched control units.
        n_unmatched: Treated units dropped due to caliper.
        balance: DataFrame of covariate balance (SMD before and after matching).
        matched_df: The matched dataset.
    """

    att: float
    ate: float
    std_err_att: float
    ci_lower_att: float
    ci_upper_att: float
    p_value_att: float
    n_treated: float
    n_matched_control: float
    n_unmatched: int
    balance: pd.DataFrame
    matched_df: pd.DataFrame

    def __str__(self) -> str:
        return (
            f"PSM ATT = {self.att:.4f} [{self.ci_lower_att:.4f}, {self.ci_upper_att:.4f}] "
            f"(p={self.p_value_att:.4f})\n"
            f"  ATE = {self.ate:.4f}\n"
            f"  Matched: {int(self.n_treated)} treated → {int(self.n_matched_control)} controls "
            f"({self.n_unmatched} unmatched due to caliper)"
        )


class PropensityScoreMatching:
    """Propensity Score Matching estimator.

    Args:
        n_neighbors: Number of control matches per treated unit (default 1).
        caliper: Maximum allowed difference in propensity score for a match.
            If None, no caliper is applied (all treated units are matched).
            A common rule of thumb is ``0.2 * std(logit(pscore))``.
        alpha: Significance level for confidence intervals (default 0.05).
        random_state: Random seed for logistic regression (default 42).

    Example::

        psm = PropensityScoreMatching(caliper=0.05)
        result = psm.match(
            df=df,
            treatment_col="is_treated",
            outcome_col="revenue",
            covariate_cols=["age", "tenure_days", "plan_growth"],
        )
        print(result)
        print(result.balance)
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        caliper: float | None = None,
        alpha: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.caliper = caliper
        self.alpha = alpha
        self.random_state = random_state

    def match(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: list[str],
    ) -> MatchingResult:
        """Estimate treatment effect via propensity score matching.

        Args:
            df: DataFrame with one row per unit.
            treatment_col: Binary column (0/1 or bool).
            outcome_col: Outcome variable column.
            covariate_cols: List of pre-treatment covariate columns.

        Returns:
            :class:`MatchingResult` with ATT, ATE, and balance diagnostics.
        """
        df = df[[treatment_col, outcome_col] + covariate_cols].dropna().copy()
        treat = df[treatment_col].astype(int).values
        y = df[outcome_col].astype(float).values
        X = df[covariate_cols].astype(float).values

        # Estimate propensity scores
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        lr.fit(X_scaled, treat)
        pscore = lr.predict_proba(X_scaled)[:, 1]
        df["_pscore"] = pscore

        # Match treated → control via nearest neighbour on propensity score
        treated_idx = np.where(treat == 1)[0]
        control_idx = np.where(treat == 0)[0]

        tree = cKDTree(pscore[control_idx].reshape(-1, 1))
        dists, matched_positions = tree.query(
            pscore[treated_idx].reshape(-1, 1),
            k=min(self.n_neighbors, len(control_idx)),
        )

        # Apply caliper
        if self.n_neighbors == 1:
            dists = dists.reshape(-1, 1)
            matched_positions = matched_positions.reshape(-1, 1)

        valid_mask = np.ones(len(treated_idx), dtype=bool)
        if self.caliper is not None:
            valid_mask = (dists.min(axis=1) <= self.caliper)

        n_unmatched = int((~valid_mask).sum())
        if n_unmatched > 0:
            logger.warning("%d treated units unmatched (caliper=%.4f)", n_unmatched, self.caliper)

        matched_treated_idx = treated_idx[valid_mask]
        matched_control_idx = control_idx[matched_positions[valid_mask].flatten()]

        # ATT: mean outcome difference within matched pairs
        att_diffs = y[matched_treated_idx] - y[matched_control_idx[:len(matched_treated_idx)]]
        att = float(att_diffs.mean())
        se_att = float(att_diffs.std(ddof=1) / np.sqrt(len(att_diffs))) if len(att_diffs) > 1 else float("nan")
        t_crit = scipy_stats.t.ppf(1 - self.alpha / 2, df=max(len(att_diffs) - 1, 1))
        p_value = float(2 * scipy_stats.t.sf(abs(att / se_att), df=len(att_diffs) - 1)) if se_att > 0 else float("nan")

        # ATE via inverse probability weighting (IPW) as a supplement
        eps = 1e-8
        ipw_weights = treat / (pscore + eps) - (1 - treat) / (1 - pscore + eps)
        ate = float(np.mean(ipw_weights * y))

        # Matched DataFrame
        matched_rows = pd.concat([
            df.iloc[matched_treated_idx].assign(_matched_group="treated"),
            df.iloc[matched_control_idx[:len(matched_treated_idx)]].assign(_matched_group="control"),
        ], ignore_index=True)

        balance = self._balance_table(df, matched_rows, treatment_col, covariate_cols)

        return MatchingResult(
            att=att,
            ate=ate,
            std_err_att=se_att,
            ci_lower_att=att - t_crit * se_att,
            ci_upper_att=att + t_crit * se_att,
            p_value_att=p_value,
            n_treated=float(valid_mask.sum()),
            n_matched_control=float(len(matched_control_idx[:len(matched_treated_idx)])),
            n_unmatched=n_unmatched,
            balance=balance,
            matched_df=matched_rows,
        )

    @staticmethod
    def _balance_table(
        full_df: pd.DataFrame,
        matched_df: pd.DataFrame,
        treatment_col: str,
        covariate_cols: list[str],
    ) -> pd.DataFrame:
        """Compute Standardised Mean Difference (SMD) before and after matching.

        SMD < 0.1 is generally considered good balance.
        """
        rows = []
        for col in covariate_cols:
            t_full = full_df.loc[full_df[treatment_col] == 1, col].astype(float)
            c_full = full_df.loc[full_df[treatment_col] == 0, col].astype(float)
            t_matched = matched_df.loc[matched_df["_matched_group"] == "treated", col].astype(float)
            c_matched = matched_df.loc[matched_df["_matched_group"] == "control", col].astype(float)

            pooled_std = np.sqrt((t_full.var() + c_full.var()) / 2)
            smd_before = (t_full.mean() - c_full.mean()) / (pooled_std + 1e-10)
            smd_after = (t_matched.mean() - c_matched.mean()) / (pooled_std + 1e-10)
            rows.append({
                "covariate": col,
                "mean_treated": t_full.mean(),
                "mean_control": c_full.mean(),
                "smd_before": smd_before,
                "smd_after": smd_after,
                "balanced": abs(smd_after) < 0.1,
            })
        return pd.DataFrame(rows)
