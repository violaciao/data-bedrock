"""Tests for 05_measurement/causal_inference/mmm.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_inference.mmm import (
    MarketingMixModel,
    ChannelConfig,
    MMMResult,
    adstock,
    saturation,
)


def _make_mmm_data(n_weeks: int = 52, seed: int = 0) -> pd.DataFrame:
    """Synthetic weekly marketing data with known ground-truth contributions."""
    rng = np.random.default_rng(seed)
    spend_search = rng.uniform(500, 2000, n_weeks)
    spend_social = rng.uniform(200, 1000, n_weeks)
    spend_email = rng.uniform(50, 300, n_weeks)
    baseline = 5000 + np.linspace(0, 500, n_weeks)

    # Transform and generate revenue.
    # Coefficients are large so channel contributions (≈500-2000 each) dominate
    # noise (std=100), giving OLS enough signal to recover positive coefficients.
    search_contrib = 1500.0 * saturation(adstock(spend_search, 0.3), 2.0, 1000)
    social_contrib = 1000.0 * saturation(adstock(spend_social, 0.5), 1.5, 500)
    email_contrib = 2000.0 * saturation(adstock(spend_email, 0.1), 3.0, 150)
    revenue = baseline + search_contrib + social_contrib + email_contrib + rng.normal(0, 100, n_weeks)

    return pd.DataFrame({
        "paid_search": spend_search,
        "social": spend_social,
        "email": spend_email,
        "trend": np.linspace(0, 1, n_weeks),
        "revenue": revenue,
    })


class TestAdstock:
    def test_zero_decay_returns_spend(self):
        x = np.array([100.0, 200.0, 300.0])
        assert np.allclose(adstock(x, 0.0), x)

    def test_carryover_increases_values(self):
        x = np.array([100.0, 0.0, 0.0])
        result = adstock(x, 0.5)
        assert result[0] == 100.0
        assert result[1] == pytest.approx(50.0)
        assert result[2] == pytest.approx(25.0)

    def test_invalid_decay_raises(self):
        with pytest.raises(ValueError):
            adstock(np.array([1.0, 2.0]), decay=1.5)

    def test_output_shape(self):
        x = np.random.rand(20)
        assert adstock(x, 0.3).shape == x.shape


class TestSaturation:
    def test_zero_input_gives_zero(self):
        assert saturation(np.array([0.0]), alpha=2.0, K=100.0)[0] == 0.0

    def test_output_between_zero_and_one(self):
        x = np.linspace(0, 10_000, 100)
        y = saturation(x, alpha=2.0, K=500.0)
        assert (y >= 0).all() and (y < 1.0).all()

    def test_half_saturation_at_K(self):
        K = 300.0
        y = saturation(np.array([K]), alpha=2.0, K=K)
        assert y[0] == pytest.approx(0.5)

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            saturation(np.array([1.0]), alpha=-1.0, K=100.0)


class TestMarketingMixModel:
    @pytest.fixture
    def df(self):
        return _make_mmm_data()

    @pytest.fixture
    def mmm(self):
        return MarketingMixModel(
            channel_configs=[
                ChannelConfig("paid_search", decay=0.3, alpha=2.0, K=1000),
                ChannelConfig("social",      decay=0.5, alpha=1.5, K=500),
                ChannelConfig("email",       decay=0.1, alpha=3.0, K=150),
            ],
            control_cols=["trend"],
        )

    def test_fit_returns_mmm_result(self, mmm, df):
        result = mmm.fit(df, "revenue")
        assert isinstance(result, MMMResult)

    def test_r_squared_high(self, mmm, df):
        result = mmm.fit(df, "revenue")
        assert result.r_squared > 0.80

    def test_channel_coefficients_positive(self, mmm, df):
        result = mmm.fit(df, "revenue")
        for coef in result.coefficients.values():
            assert coef > 0

    def test_contribution_shares_sum_to_less_than_one(self, mmm, df):
        result = mmm.fit(df, "revenue")
        total_share = result.channel_summary["contribution_share"].sum()
        assert 0.0 < total_share <= 1.0

    def test_roi_positive(self, mmm, df):
        result = mmm.fit(df, "revenue")
        assert (result.channel_summary["roi"] > 0).all()

    def test_response_curve_returns_dataframe(self, mmm, df):
        mmm.fit(df, "revenue")
        rc = mmm.response_curve("paid_search")
        assert "spend" in rc.columns
        assert "predicted_contribution" in rc.columns
        assert (rc["predicted_contribution"] >= 0).all()

    def test_response_curve_before_fit_raises(self, mmm):
        with pytest.raises(RuntimeError):
            mmm.response_curve("paid_search")

    def test_optimize_budget_sums_to_budget(self, mmm, df):
        mmm.fit(df, "revenue")
        budget = 5000.0
        plan = mmm.optimize_budget(total_budget=budget, weeks=1)
        assert plan["optimised_spend"].sum() == pytest.approx(budget, rel=0.01)

    def test_default_channel_configs(self, df):
        # No explicit configs: should auto-detect channels
        mmm = MarketingMixModel(control_cols=["trend"])
        result = mmm.fit(df, "revenue")
        assert len(result.coefficients) == 3  # paid_search, social, email

    def test_str_representation(self, mmm, df):
        result = mmm.fit(df, "revenue")
        s = str(result)
        assert "R²" in s
        assert "paid_search" in s
