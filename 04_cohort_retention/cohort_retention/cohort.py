"""Cohort analysis: build the cohort × period retention matrix.

The matrix produced here is the standard input for retention heatmaps and
survival curve plots. Each cell represents the fraction of a cohort that
was still active in a given period after their first activity.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_cohort_matrix(
    events: pd.DataFrame,
    user_col: str = "user_id",
    date_col: str = "occurred_at",
    period: str = "W",
    max_periods: int = 26,
) -> pd.DataFrame:
    """Build a cohort retention matrix from event-level data.

    Args:
        events: DataFrame with at least ``user_col`` and ``date_col`` columns.
        user_col: Column identifying the user.
        date_col: Column with the event timestamp (will be coerced to datetime).
        period: Pandas offset alias for cohort bucketing. ``"W"`` for weekly,
            ``"M"`` for monthly (default ``"W"``).
        max_periods: Maximum number of periods to include (default 26 = 6 months weekly).

    Returns:
        DataFrame indexed by ``cohort_period`` (first activity bucket) with
        columns ``0, 1, 2, …, max_periods-1`` containing retention rates (0–1).
        Period 0 is always 1.0 by definition.

    Example::

        matrix = build_cohort_matrix(events_df)
        # matrix.columns = [0, 1, 2, ..., 25]
        # matrix.index   = ['2024-W01', '2024-W02', ...]
    """
    df = events[[user_col, date_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Assign each event to a period bucket
    df["event_period"] = df[date_col].dt.to_period(period)

    # Each user's cohort = the first period they appeared
    cohort_map = (
        df.groupby(user_col)["event_period"]
        .min()
        .rename("cohort_period")
    )
    df = df.join(cohort_map, on=user_col)

    # Period offset relative to cohort
    df["period_offset"] = (df["event_period"] - df["cohort_period"]).apply(
        lambda x: x.n if hasattr(x, "n") else int(x)
    )

    # Count distinct active users per (cohort, offset)
    active = (
        df[df["period_offset"].between(0, max_periods - 1)]
        .groupby(["cohort_period", "period_offset"])[user_col]
        .nunique()
        .reset_index()
        .rename(columns={user_col: "active_users"})
    )

    # Cohort sizes (period 0)
    cohort_sizes = active[active["period_offset"] == 0].set_index("cohort_period")["active_users"]

    # Pivot to wide format
    matrix = active.pivot(index="cohort_period", columns="period_offset", values="active_users")
    matrix = matrix.reindex(columns=range(max_periods))

    # Normalise by cohort size to get retention rates
    matrix = matrix.div(cohort_sizes, axis=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns.name = "periods_since_first_activity"

    logger.info(
        "Cohort matrix: %d cohorts × %d periods, avg period-1 retention=%.1f%%",
        len(matrix),
        max_periods,
        matrix[1].mean() * 100 if 1 in matrix.columns else float("nan"),
    )
    return matrix


def cohort_sizes(
    events: pd.DataFrame,
    user_col: str = "user_id",
    date_col: str = "occurred_at",
    period: str = "W",
) -> pd.Series:
    """Return the number of new users per cohort period.

    Args:
        events: Event-level DataFrame.
        user_col: User identifier column.
        date_col: Timestamp column.
        period: Pandas period alias.

    Returns:
        Series indexed by period string with new-user counts.
    """
    df = events[[user_col, date_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    first_seen = df.groupby(user_col)[date_col].min().dt.to_period(period)
    return first_seen.value_counts().sort_index().rename("cohort_size")
