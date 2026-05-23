"""Acquisition channel analytics."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def acquisition_metrics(
    users: pd.DataFrame,
    orders: pd.DataFrame | None = None,
    user_col: str = "user_id",
    channel_col: str = "acquisition_channel",
    converted_col: str = "converted",
    signup_col: str = "signup_at",
    amount_col: str = "amount_usd",
) -> pd.DataFrame:
    """Compute acquisition metrics per channel.

    Args:
        users: Users DataFrame.
        orders: Optional orders DataFrame. If provided, revenue metrics are
            included.
        user_col: User identifier column.
        channel_col: Acquisition channel column.
        converted_col: Boolean column indicating paid conversion.
        signup_col: Signup timestamp column.
        amount_col: Revenue amount column in orders (if provided).

    Returns:
        DataFrame indexed by channel with columns:
        - ``signups``: total signups
        - ``conversions``: number of paid conversions
        - ``conversion_rate``: conversions / signups
        - ``total_revenue``: sum of all orders (if orders provided)
        - ``arpu``: average revenue per converted user (if orders provided)
    """
    users = users.copy()
    users[signup_col] = pd.to_datetime(users[signup_col])

    metrics = (
        users.groupby(channel_col)
        .agg(
            signups=(user_col, "count"),
            conversions=(converted_col, "sum"),
        )
        .reset_index()
    )
    metrics["conversion_rate"] = metrics["conversions"] / metrics["signups"]

    if orders is not None:
        revenue = (
            orders.groupby(user_col)[amount_col]
            .sum()
            .reset_index()
            .rename(columns={amount_col: "total_revenue"})
        )
        user_channel = users[[user_col, channel_col]]
        revenue_by_channel = (
            revenue.merge(user_channel, on=user_col)
            .groupby(channel_col)["total_revenue"]
            .sum()
            .reset_index()
        )
        metrics = metrics.merge(revenue_by_channel, on=channel_col, how="left")
        metrics["arpu"] = metrics["total_revenue"] / metrics["conversions"].replace(0, float("nan"))

    metrics = metrics.sort_values("signups", ascending=False).reset_index(drop=True)
    logger.info("Acquisition metrics computed for %d channels.", len(metrics))
    return metrics


def channel_comparison(
    users: pd.DataFrame,
    channel_col: str = "acquisition_channel",
    converted_col: str = "converted",
    signup_col: str = "signup_at",
    period: str = "M",
) -> pd.DataFrame:
    """Monthly (or other period) breakdown of signups and conversions by channel.

    Args:
        users: Users DataFrame.
        channel_col: Channel column.
        converted_col: Conversion boolean column.
        signup_col: Signup timestamp column.
        period: Pandas period alias for time bucketing (default ``"M"``).

    Returns:
        DataFrame with columns: period, channel, signups, conversions,
        conversion_rate — suitable for a time-series line chart.
    """
    df = users.copy()
    df[signup_col] = pd.to_datetime(df[signup_col])
    df["period"] = df[signup_col].dt.to_period(period).astype(str)

    result = (
        df.groupby(["period", channel_col])
        .agg(
            signups=("user_id", "count"),
            conversions=(converted_col, "sum"),
        )
        .reset_index()
    )
    result["conversion_rate"] = result["conversions"] / result["signups"]
    return result.sort_values(["period", channel_col]).reset_index(drop=True)
