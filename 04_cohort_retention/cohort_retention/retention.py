"""Retention curve fitting and channel-level retention comparisons."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retention curve models
# ---------------------------------------------------------------------------


def _power_law(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power-law decay: r(t) = a * t^(-b)."""
    return a * np.power(t + 1, -b)


def _exponential_decay(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """Exponential decay: r(t) = a * exp(-b * t)."""
    return a * np.exp(-b * t)


MODEL_REGISTRY: dict[str, Callable] = {
    "power_law": _power_law,
    "exponential": _exponential_decay,
}


def fit_retention_curve(
    retention_series: pd.Series,
    model: str = "power_law",
) -> dict[str, float | str | np.ndarray]:
    """Fit a parametric model to a retention curve.

    Args:
        retention_series: Series where the index is the period number (0, 1, 2 …)
            and values are retention rates (0–1). Period 0 is typically 1.0.
        model: ``"power_law"`` or ``"exponential"`` (default ``"power_law"``).
            Power law fits most SaaS retention curves better than exponential
            because retention tends to stabilise at a non-zero long-run rate.

    Returns:
        Dict with keys:
        - ``model``: model name
        - ``params``: fitted parameter array (a, b)
        - ``fitted``: predicted retention at each observed period
        - ``r_squared``: goodness of fit
        - ``long_run_estimate``: predicted retention at period 52 (1 year)

    Raises:
        ValueError: If ``model`` is not recognised.
    """
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(MODEL_REGISTRY)}")

    retention_series = retention_series.dropna()
    # Exclude period 0 (always 1.0 by definition — would skew the fit)
    series = retention_series[retention_series.index > 0]
    t = series.index.to_numpy(dtype=float)
    y = series.values

    model_fn = MODEL_REGISTRY[model]
    try:
        popt, _ = curve_fit(model_fn, t, y, p0=[0.5, 0.3], bounds=(0, np.inf), maxfev=5000)
    except RuntimeError:
        logger.warning("Curve fitting did not converge for model=%s, returning NaN params.", model)
        popt = np.array([float("nan"), float("nan")])

    fitted = model_fn(t, *popt)
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    long_run = float(model_fn(np.array([52.0]), *popt)[0])

    return {
        "model": model,
        "params": popt,
        "fitted": fitted,
        "r_squared": float(r_sq),
        "long_run_estimate": long_run,
    }


def retention_by_channel(
    users: pd.DataFrame,
    events: pd.DataFrame,
    user_col: str = "user_id",
    date_col: str = "occurred_at",
    channel_col: str = "acquisition_channel",
    period: str = "W",
    periods: int = 8,
) -> pd.DataFrame:
    """Compute average retention by acquisition channel.

    Args:
        users: Users DataFrame with ``user_col`` and ``channel_col``.
        events: Events DataFrame with ``user_col`` and ``date_col``.
        user_col: User identifier column.
        date_col: Event timestamp column.
        channel_col: Acquisition channel column in users table.
        period: Period alias for retention bucketing.
        periods: Number of periods to include (default 8).

    Returns:
        DataFrame with columns ``[channel, 0, 1, ..., periods-1]`` where
        each number column is the mean retention rate at that period offset.
    """
    from .cohort import build_cohort_matrix

    # Build per-channel retention matrices and average across cohorts
    channel_retentions = []
    for channel, grp in users.groupby(channel_col):
        channel_events = events[events[user_col].isin(grp[user_col])]
        if len(channel_events) < 10:
            continue
        matrix = build_cohort_matrix(channel_events, user_col=user_col, date_col=date_col, period=period, max_periods=periods)
        avg_retention = matrix.mean(axis=0).to_frame().T
        avg_retention.insert(0, "channel", channel)
        channel_retentions.append(avg_retention)

    if not channel_retentions:
        return pd.DataFrame()

    return pd.concat(channel_retentions, ignore_index=True)
