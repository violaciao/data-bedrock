"""Tests for 05_ab_testing/src/sample_size.py."""

from __future__ import annotations

import pytest

from ab_testing.sample_size import SampleSizeResult, required_sample_size


class TestRequiredSampleSize:
    def test_basic_relative_mde(self):
        result = required_sample_size(baseline_rate=0.05, mde_relative=0.20)
        assert isinstance(result, SampleSizeResult)
        assert result.n_per_variant > 0
        assert result.n_total == result.n_per_variant * 2

    def test_basic_absolute_mde(self):
        result = required_sample_size(baseline_rate=0.05, mde_absolute=0.01)
        assert result.n_per_variant > 0

    def test_larger_mde_needs_smaller_n(self):
        small_mde = required_sample_size(baseline_rate=0.10, mde_relative=0.05)
        large_mde = required_sample_size(baseline_rate=0.10, mde_relative=0.20)
        assert small_mde.n_per_variant > large_mde.n_per_variant

    def test_higher_power_needs_larger_n(self):
        low = required_sample_size(baseline_rate=0.10, mde_relative=0.10, power=0.80)
        high = required_sample_size(baseline_rate=0.10, mde_relative=0.10, power=0.90)
        assert high.n_per_variant > low.n_per_variant

    def test_lower_alpha_needs_larger_n(self):
        loose = required_sample_size(baseline_rate=0.10, mde_relative=0.10, alpha=0.10)
        strict = required_sample_size(baseline_rate=0.10, mde_relative=0.10, alpha=0.01)
        assert strict.n_per_variant > loose.n_per_variant

    def test_estimated_days_returned(self):
        result = required_sample_size(
            baseline_rate=0.05,
            mde_relative=0.20,
            daily_traffic_per_variant=1000,
        )
        assert result.estimated_days is not None
        assert result.estimated_days == pytest.approx(result.n_per_variant / 1000, abs=1)

    def test_no_daily_traffic_no_duration(self):
        result = required_sample_size(baseline_rate=0.05, mde_relative=0.20)
        assert result.estimated_days is None

    def test_both_mde_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            required_sample_size(baseline_rate=0.05, mde_relative=0.10, mde_absolute=0.01)

    def test_neither_mde_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            required_sample_size(baseline_rate=0.05)

    def test_invalid_baseline_rate(self):
        with pytest.raises(ValueError, match="baseline_rate"):
            required_sample_size(baseline_rate=1.5, mde_relative=0.10)

    def test_n_variants_scales_total(self):
        result = required_sample_size(baseline_rate=0.05, mde_relative=0.20, n_variants=3)
        assert result.n_total == result.n_per_variant * 3

    def test_mde_absolute_stored_correctly(self):
        result = required_sample_size(baseline_rate=0.10, mde_absolute=0.02)
        assert result.mde_absolute == pytest.approx(0.02)

    def test_mde_relative_converted_to_absolute(self):
        result = required_sample_size(baseline_rate=0.10, mde_relative=0.20)
        assert result.mde_absolute == pytest.approx(0.02, rel=1e-6)
