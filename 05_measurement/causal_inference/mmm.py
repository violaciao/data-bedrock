"""Marketing Mix Modeling (MMM).

MMM estimates the contribution of each marketing channel to revenue (or any
outcome) using regression with two key transformations:

1. **Adstock / Carryover** — marketing has a delayed and decaying effect.
   Spend today influences sales this week AND in future weeks.
   Modelled as a geometric decay: adstock_t = spend_t + λ · adstock_{t-1}

2. **Saturation / Diminishing returns** — doubling spend does not double
   the outcome. Modelled with a Hill function:
   f(x) = x^α / (K^α + x^α)

After transformation, the model is a simple linear regression. This keeps
it interpretable and auditable — the gold standard for budget decisions.

This module provides:
- Adstock and saturation transformations
- Model fitting via OLS
- Channel contribution decomposition
- Response curve plots
- Simple budget optimisation (gradient-free)

Design decisions:
- We use OLS with hand-crafted feature transformations rather than Bayesian
  MMM (e.g. PyMC-Marketing) because it requires no additional dependencies
  and is fully transparent. For production use with uncertainty quantification,
  consider switching to a Bayesian implementation.
- Parameters (lambda, alpha, K) are treated as hyperparameters tuned by
  grid search over a validation window, not inferred jointly with the
  regression coefficients.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


def adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Apply geometric adstock (carryover) transformation.

    Args:
        spend: 1-D array of weekly spend values.
        decay: Decay rate λ ∈ [0, 1). 0 = no carryover, 0.9 = long tail.

    Returns:
        1-D array of adstocked spend.
    """
    if not 0 <= decay < 1:
        raise ValueError(f"decay must be in [0, 1), got {decay}")
    result = np.empty_like(spend, dtype=float)
    result[0] = spend[0]
    for t in range(1, len(spend)):
        result[t] = spend[t] + decay * result[t - 1]
    return result


def saturation(x: np.ndarray, alpha: float, K: float) -> np.ndarray:
    """Apply Hill saturation (diminishing returns) transformation.

    f(x) = x^α / (K^α + x^α)

    Args:
        x: Input values (e.g. adstocked spend).
        alpha: Shape parameter α > 0. Higher = steeper S-curve.
        K: Half-saturation point. Spend at which f(x) = 0.5.

    Returns:
        Transformed values in [0, 1).
    """
    if alpha <= 0 or K <= 0:
        raise ValueError("alpha and K must be > 0")
    x = np.asarray(x, dtype=float)
    return x ** alpha / (K ** alpha + x ** alpha)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class ChannelConfig:
    """Transformation parameters for a single marketing channel.

    Attributes:
        name: Channel name (must match a column in the spend DataFrame).
        decay: Adstock decay rate λ ∈ [0, 1).
        alpha: Hill saturation shape parameter.
        K: Hill half-saturation point.
    """

    name: str
    decay: float = 0.5
    alpha: float = 2.0
    K: float = 1.0


@dataclass
class MMMResult:
    """Fitted MMM result.

    Attributes:
        coefficients: Dict mapping channel name → regression coefficient.
        intercept: Regression intercept.
        r_squared: In-sample R².
        contributions: DataFrame with weekly contribution per channel.
        channel_summary: Aggregated contribution and ROI per channel.
    """

    coefficients: dict[str, float]
    intercept: float
    r_squared: float
    contributions: pd.DataFrame
    channel_summary: pd.DataFrame

    def __str__(self) -> str:
        lines = [f"MMM R² = {self.r_squared:.3f}", "Channel contributions:"]
        for _, row in self.channel_summary.iterrows():
            lines.append(
                f"  {row['channel']:20s}  contrib={row['total_contribution']:,.0f}  "
                f"share={row['contribution_share']:.1%}  roi={row['roi']:.2f}x"
            )
        return "\n".join(lines)


class MarketingMixModel:
    """Marketing Mix Model with adstock and saturation transformations.

    Args:
        channel_configs: List of :class:`ChannelConfig` objects, one per
            marketing channel. If None, defaults to adstock=0.5, alpha=2, K=1
            for each column passed to :meth:`fit`.
        control_cols: Column names for non-marketing predictors (e.g. seasonality,
            price index, trend). These are included in the regression as-is
            (no adstock/saturation transformation).

    Example::

        mmm = MarketingMixModel(
            channel_configs=[
                ChannelConfig("paid_search", decay=0.3, alpha=2.0, K=500),
                ChannelConfig("social",      decay=0.6, alpha=1.5, K=300),
                ChannelConfig("email",       decay=0.1, alpha=3.0, K=100),
            ],
            control_cols=["trend", "seasonality_index"],
        )
        result = mmm.fit(df, outcome_col="revenue")
        print(result)
        budget_plan = mmm.optimize_budget(total_budget=10_000, weeks=4)
    """

    def __init__(
        self,
        channel_configs: list[ChannelConfig] | None = None,
        control_cols: list[str] | None = None,
    ) -> None:
        self.channel_configs = channel_configs
        self.control_cols = control_cols or []
        self._result: MMMResult | None = None
        self._channel_names: list[str] = []
        self._df_fit: pd.DataFrame | None = None
        self._outcome_col: str = ""

    def fit(self, df: pd.DataFrame, outcome_col: str) -> MMMResult:
        """Fit the MMM to historical data.

        Args:
            df: DataFrame with one row per time period (week). Must contain
                columns for each channel and any control variables.
            outcome_col: The outcome variable to model (e.g. ``"revenue"``).

        Returns:
            :class:`MMMResult` with coefficients, R², and contributions.
        """
        df = df.copy()
        y = df[outcome_col].astype(float).values

        # Default channel configs: one per non-outcome, non-control column
        if self.channel_configs is None:
            channel_cols = [c for c in df.columns if c != outcome_col and c not in self.control_cols]
            self.channel_configs = [ChannelConfig(name=c) for c in channel_cols]

        self._channel_names = [cfg.name for cfg in self.channel_configs]
        self._df_fit = df
        self._outcome_col = outcome_col

        # Apply transformations
        X_parts = [np.ones(len(df))]
        for cfg in self.channel_configs:
            x = df[cfg.name].astype(float).values
            x_adstock = adstock(x, cfg.decay)
            x_sat = saturation(x_adstock, cfg.alpha, cfg.K)
            X_parts.append(x_sat)

        # Add control variables untransformed
        for col in self.control_cols:
            X_parts.append(df[col].astype(float).values)

        X = np.column_stack(X_parts)

        # OLS
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ coef
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        intercept = float(coef[0])
        channel_coefs = {cfg.name: float(coef[i + 1]) for i, cfg in enumerate(self.channel_configs)}

        # Contributions: coef_i × transformed_spend_i
        contrib_cols = {}
        for i, cfg in enumerate(self.channel_configs):
            x = df[cfg.name].astype(float).values
            x_transformed = saturation(adstock(x, cfg.decay), cfg.alpha, cfg.K)
            contrib_cols[cfg.name] = channel_coefs[cfg.name] * x_transformed

        contributions = pd.DataFrame(contrib_cols)
        contributions["baseline"] = intercept
        contributions["fitted"] = y_hat
        contributions["actual"] = y
        if "date" in df.columns:
            contributions.insert(0, "date", df["date"].values)

        # Channel summary
        total_revenue = float(y.sum())
        rows = []
        for cfg in self.channel_configs:
            total_contrib = float(contributions[cfg.name].sum())
            total_spend = float(df[cfg.name].sum())
            rows.append({
                "channel": cfg.name,
                "total_contribution": total_contrib,
                "contribution_share": total_contrib / total_revenue if total_revenue else float("nan"),
                "total_spend": total_spend,
                "roi": total_contrib / total_spend if total_spend > 0 else float("nan"),
            })
        channel_summary = pd.DataFrame(rows).sort_values("total_contribution", ascending=False)

        self._result = MMMResult(
            coefficients=channel_coefs,
            intercept=intercept,
            r_squared=r_sq,
            contributions=contributions,
            channel_summary=channel_summary,
        )
        logger.info("MMM fitted: R²=%.3f  channels=%s", r_sq, self._channel_names)
        return self._result

    def response_curve(
        self,
        channel: str,
        spend_range: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Compute the response curve for a channel (diminishing returns).

        Args:
            channel: Channel name.
            spend_range: Array of spend values to evaluate. Defaults to
                0 → 2× historical max spend for the channel.

        Returns:
            DataFrame with columns: spend, transformed_spend, predicted_contribution.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before response_curve().")
        cfg = next(c for c in self.channel_configs if c.name == channel)  # type: ignore[union-attr]
        coef = self._result.coefficients[channel]

        if spend_range is None:
            max_spend = float(self._df_fit[channel].max()) * 2  # type: ignore[index]
            spend_range = np.linspace(0, max_spend, 200)

        transformed = saturation(adstock(spend_range, cfg.decay), cfg.alpha, cfg.K)
        return pd.DataFrame({
            "spend": spend_range,
            "transformed_spend": transformed,
            "predicted_contribution": coef * transformed,
        })

    def optimize_budget(
        self,
        total_budget: float,
        weeks: int = 1,
    ) -> pd.DataFrame:
        """Simple budget allocation optimisation to maximise predicted contribution.

        Uses Scipy's L-BFGS-B to find the spend split across channels that
        maximises total predicted contribution given a budget constraint.
        Each channel spend is bounded to [0, total_budget].

        Args:
            total_budget: Total budget to allocate across channels.
            weeks: Number of weeks the budget covers (spend is split evenly).

        Returns:
            DataFrame with columns: channel, optimised_spend, predicted_contribution, roi.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before optimize_budget().")

        n = len(self.channel_configs)  # type: ignore[arg-type]
        weekly_budget = total_budget / weeks

        def neg_contribution(alloc: np.ndarray) -> float:
            total = 0.0
            for i, cfg in enumerate(self.channel_configs):  # type: ignore[union-attr]
                coef = self._result.coefficients[cfg.name]  # type: ignore[union-attr]
                x_t = saturation(adstock(np.full(weeks, alloc[i]), cfg.decay), cfg.alpha, cfg.K)
                total += coef * x_t.sum()
            return -total

        x0 = np.full(n, weekly_budget / n)
        bounds = [(0, weekly_budget)] * n
        constraints = [{"type": "ineq", "fun": lambda x: weekly_budget - x.sum()}]
        res = minimize(neg_contribution, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        rows = []
        for i, cfg in enumerate(self.channel_configs):  # type: ignore[union-attr]
            spend = float(res.x[i]) * weeks
            coef = self._result.coefficients[cfg.name]  # type: ignore[union-attr]
            x_t = saturation(adstock(np.full(weeks, float(res.x[i])), cfg.decay), cfg.alpha, cfg.K)
            contrib = float(coef * x_t.sum())
            rows.append({
                "channel": cfg.name,
                "optimised_spend": spend,
                "predicted_contribution": contrib,
                "roi": contrib / spend if spend > 0 else float("nan"),
            })
        return pd.DataFrame(rows).sort_values("optimised_spend", ascending=False)
