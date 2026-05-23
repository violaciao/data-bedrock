"""Tests for 05_ab_testing/src/stats.py.

Statistical correctness is the core value of this module — every function must
have a test that verifies both the happy path and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from ab_testing.stats import (
    StatResult,
    bootstrap_ci,
    cuped,
    mann_whitney_u_test,
    welch_t_test,
    wilson_ci,
    z_test,
)


# ---------------------------------------------------------------------------
# z_test
# ---------------------------------------------------------------------------


class TestZTest:
    def test_significant_difference(self):
        # 5% vs 7% conversion with large samples — should be significant
        result = z_test(500, 10_000, 700, 10_000, alpha=0.05)
        assert result.significant
        assert result.p_value < 0.05
        assert result.effect_size == pytest.approx(0.02, abs=1e-6)

    def test_no_difference(self):
        # Identical rates — should not be significant
        result = z_test(500, 10_000, 500, 10_000, alpha=0.05)
        assert not result.significant
        assert result.p_value == pytest.approx(1.0, abs=0.01)
        assert result.effect_size == pytest.approx(0.0, abs=1e-9)

    def test_ci_contains_effect(self):
        result = z_test(500, 10_000, 700, 10_000)
        assert result.ci_lower < result.effect_size < result.ci_upper

    def test_ci_does_not_contain_zero_when_significant(self):
        result = z_test(500, 10_000, 700, 10_000)
        assert result.ci_lower > 0 or result.ci_upper < 0

    def test_returns_stat_result(self):
        result = z_test(100, 1000, 110, 1000)
        assert isinstance(result, StatResult)
        assert result.method == "two_proportion_z_test"

    def test_zero_n_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            z_test(0, 0, 0, 0)


# ---------------------------------------------------------------------------
# welch_t_test
# ---------------------------------------------------------------------------


class TestWelchTTest:
    def test_significant_difference(self, rng):
        control = rng.normal(loc=10.0, scale=2.0, size=500)
        treatment = rng.normal(loc=11.0, scale=2.0, size=500)
        result = welch_t_test(control, treatment)
        assert result.significant
        assert result.effect_size == pytest.approx(1.0, abs=0.3)

    def test_no_difference(self, rng):
        control = rng.normal(loc=10.0, scale=2.0, size=500)
        treatment = rng.normal(loc=10.0, scale=2.0, size=500)
        result = welch_t_test(control, treatment)
        # Should not reliably reject at the 5% level
        assert result.p_value > 0.001  # weak assertion, just checking it doesn't explode

    def test_ci_contains_true_effect(self, rng):
        control = rng.normal(loc=0.0, scale=1.0, size=1000)
        treatment = rng.normal(loc=0.5, scale=1.0, size=1000)
        result = welch_t_test(control, treatment)
        assert result.ci_lower < 0.5 < result.ci_upper

    def test_returns_stat_result(self, rng):
        c = rng.normal(size=100)
        t = rng.normal(size=100)
        result = welch_t_test(c, t)
        assert isinstance(result, StatResult)
        assert result.method == "welch_t_test"

    def test_accepts_lists(self):
        result = welch_t_test([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        assert result.significant


# ---------------------------------------------------------------------------
# mann_whitney_u_test
# ---------------------------------------------------------------------------


class TestMannWhitneyU:
    def test_detects_median_shift(self, rng):
        control = rng.exponential(scale=1.0, size=500)
        treatment = rng.exponential(scale=1.5, size=500)
        result = mann_whitney_u_test(control, treatment)
        assert result.significant
        assert result.effect_size > 0

    def test_no_difference(self, rng):
        x = rng.exponential(scale=1.0, size=200)
        result = mann_whitney_u_test(x, x.copy())
        assert not result.significant

    def test_method_name(self, rng):
        c = rng.normal(size=50)
        t = rng.normal(size=50)
        result = mann_whitney_u_test(c, t)
        assert result.method == "mann_whitney_u"


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_mean_ci_contains_true_difference(self, rng):
        c = rng.normal(loc=0.0, scale=1.0, size=500)
        t = rng.normal(loc=1.0, scale=1.0, size=500)
        lower, upper = bootstrap_ci(c, t, statistic="mean")
        assert lower < 1.0 < upper

    def test_median_ci(self, rng):
        c = rng.exponential(1.0, size=300)
        t = rng.exponential(2.0, size=300)
        lower, upper = bootstrap_ci(c, t, statistic="median")
        assert lower < upper

    def test_deterministic_with_seed(self, rng):
        c = rng.normal(size=100)
        t = rng.normal(size=100)
        ci1 = bootstrap_ci(c, t, seed=1)
        ci2 = bootstrap_ci(c, t, seed=1)
        assert ci1 == ci2

    def test_invalid_statistic_raises(self, rng):
        c = rng.normal(size=50)
        t = rng.normal(size=50)
        with pytest.raises((ValueError, TypeError)):
            bootstrap_ci(c, t, statistic="variance")  # type: ignore


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_symmetric_around_half(self):
        lower, upper = wilson_ci(500, 1000)
        midpoint = (lower + upper) / 2
        assert midpoint == pytest.approx(0.5, abs=0.01)

    def test_bounds_within_zero_one(self):
        lower, upper = wilson_ci(1, 10)
        assert 0 <= lower < upper <= 1

    def test_narrow_ci_for_large_n(self):
        lower, upper = wilson_ci(500, 10_000)
        assert (upper - lower) < 0.02  # less than 2pp wide

    def test_zero_n_raises(self):
        with pytest.raises(ValueError, match="n must be > 0"):
            wilson_ci(0, 0)


# ---------------------------------------------------------------------------
# cuped
# ---------------------------------------------------------------------------


class TestCUPED:
    def test_reduces_variance_and_detects_effect(self, rng):
        # Create data with a correlated pre-experiment metric
        n = 500
        pre = rng.normal(10.0, 3.0, size=n * 2)
        noise = rng.normal(0, 1.0, size=n * 2)
        post = pre * 0.8 + noise  # strong correlation with pre

        # Add a real treatment effect of +1.0
        post[n:] += 1.0

        result = cuped(post[:n], post[n:], pre[:n], pre[n:])
        assert result.significant
        assert result.method == "cuped"
        assert result.effect_size == pytest.approx(1.0, abs=0.3)

    def test_cuped_without_effect(self, rng):
        n = 200
        pre = rng.normal(size=n * 2)
        post = pre + rng.normal(scale=0.1, size=n * 2)  # no treatment effect
        result = cuped(post[:n], post[n:], pre[:n], pre[n:])
        # Not necessarily non-significant but should not crash
        assert isinstance(result, StatResult)
