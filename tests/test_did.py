"""Tests for 05_measurement/causal_inference/did.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_inference.did import DifferenceInDifferences, DiDResult


class TestDifferenceInDifferences:
    @pytest.fixture
    def did(self):
        return DifferenceInDifferences(alpha=0.05)

    def test_recovers_known_att_2x2(self, did):
        rng = np.random.default_rng(0)
        # True ATT = 5.0
        pre_c = rng.normal(10, 1, 100)
        post_c = rng.normal(11, 1, 100)   # +1 trend
        pre_t = rng.normal(10, 1, 100)
        post_t = rng.normal(16, 1, 100)   # +1 trend +5 treatment
        result = did.estimate_2x2(pre_t, post_t, pre_c, post_c)
        assert result.att == pytest.approx(5.0, abs=0.3)

    def test_zero_att_not_significant(self, did):
        rng = np.random.default_rng(1)
        x = rng.normal(10, 1, 200)
        result = did.estimate_2x2(x[:100], x[100:], x[:100], x[100:])
        assert not result.significant

    def test_returns_did_result(self, did, rng):
        c = rng.normal(10, 1, 50)
        result = did.estimate_2x2(c, c + 1, c, c + 1)
        assert isinstance(result, DiDResult)

    def test_ci_contains_att(self, did, rng):
        pre_c = rng.normal(10, 1, 100)
        post_c = rng.normal(11, 1, 100)
        pre_t = rng.normal(10, 1, 100)
        post_t = rng.normal(14, 1, 100)
        result = did.estimate_2x2(pre_t, post_t, pre_c, post_c)
        assert result.ci_lower < result.att < result.ci_upper

    def test_str_representation(self, did, rng):
        c = rng.normal(size=50)
        result = did.estimate_2x2(c, c + 2, c, c)
        s = str(result)
        assert "ATT" in s
        assert "Pre" in s

    def test_panel_recovers_att(self, did):
        rng = np.random.default_rng(2)
        n_units, n_times = 20, 20
        treatment_time = 10
        rows = []
        for unit in range(n_units):
            unit_fe = rng.normal(0, 2)
            is_treated = unit < n_units // 2
            for t in range(n_times):
                time_fe = 0.5 * t
                post = t >= treatment_time
                y = unit_fe + time_fe + (3.0 if is_treated and post else 0) + rng.normal(0, 0.5)
                rows.append({"unit": unit, "time": t, "y": y, "treated": is_treated})
        df = pd.DataFrame(rows)
        result = did.estimate_panel(df, "unit", "time", "y", "treated", treatment_time)
        assert result.att == pytest.approx(3.0, abs=0.5)
        assert result.method == "twfe_panel"

    def test_event_study_returns_dataframe(self, did):
        rng = np.random.default_rng(3)
        rows = []
        for unit in range(10):
            for t in range(12):
                rows.append({"unit": unit, "time": t, "y": rng.normal(10, 1), "treated": unit < 5})
        df = pd.DataFrame(rows)
        es = did.event_study(df, "unit", "time", "y", "treated", treatment_time=6, n_pre=3, n_post=3)
        assert "relative_period" in es.columns
        assert "att" in es.columns
        assert len(es) > 0


class TestDiDPrePostMeans:
    def test_pre_post_means_correct(self):
        did = DifferenceInDifferences()
        result = did.estimate_2x2(
            pre_treated=[10.0] * 10,
            post_treated=[15.0] * 10,
            pre_control=[10.0] * 10,
            post_control=[11.0] * 10,
        )
        assert result.pre_mean_treated == pytest.approx(10.0)
        assert result.post_mean_treated == pytest.approx(15.0)
        assert result.pre_mean_control == pytest.approx(10.0)
        assert result.post_mean_control == pytest.approx(11.0)
        assert result.att == pytest.approx(4.0, abs=0.01)
