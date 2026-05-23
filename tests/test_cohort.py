"""Tests for 04_cohort_retention/src/cohort.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cohort_retention.cohort import build_cohort_matrix, cohort_sizes


def _make_events(n_users: int = 100, n_periods: int = 8, seed: int = 0) -> pd.DataFrame:
    """Create a minimal synthetic event log for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2024-01-01")
    for uid in range(n_users):
        first_week = rng.integers(0, n_periods)
        n_events = rng.integers(1, 5)
        for _ in range(n_events):
            week_offset = rng.integers(first_week, n_periods)
            ts = base + pd.Timedelta(weeks=int(week_offset))
            rows.append({"user_id": f"u{uid}", "occurred_at": ts})
    return pd.DataFrame(rows)


class TestBuildCohortMatrix:
    def test_period_zero_is_one(self):
        events = _make_events()
        matrix = build_cohort_matrix(events)
        assert (matrix[0].dropna() == 1.0).all()

    def test_retention_decreases_on_average_overall(self):
        # With small random data, per-period jumps happen; check the overall
        # trend: last-half average < first-half average.
        events = _make_events(n_users=500, n_periods=10, seed=1)
        matrix = build_cohort_matrix(events, max_periods=8)
        avg = matrix.mean(axis=0).dropna()
        first_half = avg[avg.index < 4].mean()
        last_half = avg[avg.index >= 4].mean()
        assert last_half < first_half

    def test_output_columns_are_period_numbers(self):
        events = _make_events()
        matrix = build_cohort_matrix(events, max_periods=6)
        assert list(matrix.columns) == list(range(6))

    def test_values_between_zero_and_one(self):
        events = _make_events()
        matrix = build_cohort_matrix(events)
        valid = matrix.stack().dropna()  # exclude NaN cells (future periods)
        assert (valid >= 0).all() and (valid <= 1.0).all()

    def test_string_index(self):
        import pandas as pd
        events = _make_events()
        matrix = build_cohort_matrix(events)
        assert pd.api.types.is_string_dtype(matrix.index)

    def test_monthly_period(self):
        events = _make_events(n_users=300, n_periods=52)
        matrix = build_cohort_matrix(events, period="M", max_periods=6)
        assert list(matrix.columns) == list(range(6))

    def test_with_synthetic_data(self, synthetic_events):
        matrix = build_cohort_matrix(synthetic_events, max_periods=8)
        assert len(matrix) > 0
        assert matrix[0].dropna().mean() == pytest.approx(1.0, abs=0.01)


class TestCohortSizes:
    def test_returns_series(self):
        events = _make_events()
        sizes = cohort_sizes(events)
        assert isinstance(sizes, pd.Series)

    def test_total_equals_unique_users(self):
        events = _make_events(n_users=50)
        sizes = cohort_sizes(events)
        assert sizes.sum() == 50
