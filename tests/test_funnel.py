"""Tests for 03_growth_analysis/src/funnel.py."""

from __future__ import annotations

import pandas as pd
import pytest

from growth_analysis.funnel import FunnelAnalysis, FunnelStep


def _make_events() -> pd.DataFrame:
    """Minimal event log: 100 users, 3 event types in a funnel."""
    rows = []
    for i in range(100):
        rows.append({"user_id": f"u{i}", "event_type": "page_view"})
        if i < 60:  # 60% sign up
            rows.append({"user_id": f"u{i}", "event_type": "signup"})
        if i < 30:  # 30% use a feature
            rows.append({"user_id": f"u{i}", "event_type": "feature_used"})
    return pd.DataFrame(rows)


class TestFunnelAnalysis:
    @pytest.fixture
    def events(self) -> pd.DataFrame:
        return _make_events()

    @pytest.fixture
    def funnel(self) -> FunnelAnalysis:
        return FunnelAnalysis(steps=["page_view", "signup", "feature_used"])

    def test_step_count_matches_steps(self, funnel, events):
        results = funnel.compute(events)
        assert len(results) == 3

    def test_step_zero_conversion_is_one(self, funnel, events):
        results = funnel.compute(events)
        assert results[0].conversion_from_top == pytest.approx(1.0)

    def test_step_zero_users_is_top_of_funnel(self, funnel, events):
        results = funnel.compute(events)
        assert results[0].users == 100

    def test_conversion_from_top_decreases(self, funnel, events):
        results = funnel.compute(events)
        rates = [r.conversion_from_top for r in results]
        assert rates == sorted(rates, reverse=True)

    def test_dropped_users_is_correct(self, funnel, events):
        results = funnel.compute(events)
        assert results[1].dropped == 40  # 100 - 60
        assert results[2].dropped == 30  # 60 - 30

    def test_conversion_from_previous(self, funnel, events):
        results = funnel.compute(events)
        assert results[1].conversion_from_previous == pytest.approx(0.60)
        assert results[2].conversion_from_previous == pytest.approx(0.50)

    def test_to_dataframe_shape(self, funnel, events):
        results = funnel.compute(events)
        df = FunnelAnalysis.to_dataframe(results)
        assert df.shape == (3, 6)
        assert "conversion_from_top" in df.columns

    def test_empty_steps_raises(self, events):
        with pytest.raises(ValueError, match="steps must not be empty"):
            FunnelAnalysis(steps=[]).compute(events)

    def test_step_not_in_events_gives_zero_users(self, events):
        funnel = FunnelAnalysis(steps=["page_view", "nonexistent_event"])
        results = funnel.compute(events)
        assert results[1].users == 0

    def test_with_synthetic_data(self, synthetic_events):
        funnel = FunnelAnalysis(steps=["page_view", "signup", "login", "feature_used"])
        results = funnel.compute(synthetic_events)
        assert results[0].users > 0
        assert all(r.conversion_from_top <= 1.0 for r in results)
