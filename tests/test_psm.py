"""Tests for 05_measurement/causal_inference/psm.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_inference.psm import PropensityScoreMatching, MatchingResult


def _make_psm_data(n: int = 500, true_att: float = 3.0, seed: int = 0) -> pd.DataFrame:
    """Synthetic RCT-like data where true ATT is known."""
    rng = np.random.default_rng(seed)
    age = rng.normal(35, 8, n)
    tenure = rng.exponential(200, n)
    # Treatment propensity increases with age and tenure
    logit = -2 + 0.05 * age + 0.003 * tenure
    prob = 1 / (1 + np.exp(-logit))
    treated = rng.binomial(1, prob).astype(bool)
    revenue = 0.5 * age + 0.01 * tenure + true_att * treated + rng.normal(0, 2, n)
    return pd.DataFrame({"age": age, "tenure": tenure, "treated": treated, "revenue": revenue})


class TestPropensityScoreMatching:
    @pytest.fixture
    def df(self):
        return _make_psm_data()

    def test_recovers_approximate_att(self, df):
        psm = PropensityScoreMatching()
        result = psm.match(df, "treated", "revenue", ["age", "tenure"])
        # PSM with observational data: expect att near 3.0 but with noise
        assert result.att == pytest.approx(3.0, abs=1.0)

    def test_returns_matching_result(self, df):
        psm = PropensityScoreMatching()
        result = psm.match(df, "treated", "revenue", ["age", "tenure"])
        assert isinstance(result, MatchingResult)

    def test_balance_table_has_all_covariates(self, df):
        psm = PropensityScoreMatching()
        result = psm.match(df, "treated", "revenue", ["age", "tenure"])
        assert set(result.balance["covariate"]) == {"age", "tenure"}

    def test_smd_after_matching_lower_than_before(self, df):
        psm = PropensityScoreMatching()
        result = psm.match(df, "treated", "revenue", ["age", "tenure"])
        avg_before = result.balance["smd_before"].abs().mean()
        avg_after = result.balance["smd_after"].abs().mean()
        assert avg_after < avg_before

    def test_caliper_drops_unmatched(self):
        # Extreme caliper: almost nothing should match
        df = _make_psm_data(n=200)
        psm = PropensityScoreMatching(caliper=0.001)
        result = psm.match(df, "treated", "revenue", ["age", "tenure"])
        assert result.n_unmatched > 0

    def test_matched_df_has_equal_groups(self, df):
        psm = PropensityScoreMatching()
        result = psm.match(df, "treated", "revenue", ["age", "tenure"])
        n_t = (result.matched_df["_matched_group"] == "treated").sum()
        n_c = (result.matched_df["_matched_group"] == "control").sum()
        assert n_t == n_c

    def test_str_representation(self, df):
        psm = PropensityScoreMatching()
        result = psm.match(df, "treated", "revenue", ["age", "tenure"])
        s = str(result)
        assert "ATT" in s
        assert "Matched" in s

    def test_ci_bounds_are_ordered(self, df):
        psm = PropensityScoreMatching()
        result = psm.match(df, "treated", "revenue", ["age", "tenure"])
        assert result.ci_lower_att < result.att < result.ci_upper_att
