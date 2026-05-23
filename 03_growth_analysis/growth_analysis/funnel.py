"""Funnel analysis: ordered step-by-step conversion rates.

The FunnelAnalysis class accepts an ordered list of event types and computes
per-step conversion from a user-level event log. It intentionally does not
impose a time window — add one externally by filtering your events DataFrame
before passing it in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FunnelStep:
    """Result for a single funnel step.

    Attributes:
        step: Step index (0-based).
        event: Event type name.
        users: Number of users who reached this step.
        conversion_from_top: Cumulative conversion from step 0.
        conversion_from_previous: Step-over-step conversion rate.
        dropped: Users lost between the previous step and this step.
    """

    step: int
    event: str
    users: int
    conversion_from_top: float
    conversion_from_previous: float
    dropped: int


@dataclass
class FunnelAnalysis:
    """Ordered funnel analysis from event-level data.

    Users must complete steps in order to count at each step. A user is
    counted at step N only if they also completed all steps 0 … N-1.

    Args:
        steps: Ordered list of event type names defining the funnel.
        user_col: Column identifying the user in the events DataFrame.
        event_col: Column containing the event type string.
        date_col: Column with event timestamps. If provided, the *first*
            occurrence of each event per user is used; otherwise any occurrence counts.

    Example::

        funnel = FunnelAnalysis(
            steps=["page_view", "signup", "feature_used", "upgrade_clicked"]
        )
        results = funnel.compute(events_df)
        print(funnel.to_dataframe(results))
    """

    steps: list[str]
    user_col: str = "user_id"
    event_col: str = "event_type"
    date_col: str | None = "occurred_at"

    def compute(self, events: pd.DataFrame) -> list[FunnelStep]:
        """Compute funnel conversion for each step.

        Args:
            events: Event log DataFrame.

        Returns:
            List of :class:`FunnelStep` objects, one per step.
        """
        if not self.steps:
            raise ValueError("steps must not be empty.")

        cols = [self.user_col, self.event_col]
        if self.date_col and self.date_col in events.columns:
            cols.append(self.date_col)
        df = events[cols].copy()

        # Build a wide boolean table: did user X do event Y?
        event_flags: dict[str, set] = {}
        for step_event in self.steps:
            mask = df[self.event_col] == step_event
            event_flags[step_event] = set(df.loc[mask, self.user_col].unique())

        # Walk through steps, keeping only users who completed all prior steps
        active_users = None
        results: list[FunnelStep] = []
        top_count: int | None = None

        for i, step_event in enumerate(self.steps):
            step_users = event_flags[step_event]
            if active_users is None:
                active_users = step_users
            else:
                active_users = active_users & step_users

            n = len(active_users)
            if top_count is None:
                top_count = n

            conv_top = n / top_count if top_count > 0 else 0.0
            if i == 0:
                conv_prev = 1.0
                dropped = 0
            else:
                prev_n = results[i - 1].users
                conv_prev = n / prev_n if prev_n > 0 else 0.0
                dropped = prev_n - n

            results.append(
                FunnelStep(
                    step=i,
                    event=step_event,
                    users=n,
                    conversion_from_top=conv_top,
                    conversion_from_previous=conv_prev,
                    dropped=dropped,
                )
            )

        logger.info(
            "Funnel '%s → %s': top=%d  bottom=%d  overall=%.1f%%",
            self.steps[0],
            self.steps[-1],
            top_count or 0,
            results[-1].users if results else 0,
            results[-1].conversion_from_top * 100 if results else 0,
        )
        return results

    @staticmethod
    def to_dataframe(results: list[FunnelStep]) -> pd.DataFrame:
        """Convert a list of :class:`FunnelStep` to a tidy DataFrame.

        Args:
            results: Output of :meth:`compute`.

        Returns:
            DataFrame with columns: step, event, users, conversion_from_top,
            conversion_from_previous, dropped.
        """
        return pd.DataFrame(
            [
                {
                    "step": r.step,
                    "event": r.event,
                    "users": r.users,
                    "conversion_from_top": r.conversion_from_top,
                    "conversion_from_previous": r.conversion_from_previous,
                    "dropped": r.dropped,
                }
                for r in results
            ]
        )
