"""Tests for 05_ab_testing/src/sequential.py."""

from __future__ import annotations

import numpy as np
import pytest

from ab_testing.sequential import SequentialResult, SequentialTest


class TestSequentialTest:
    def test_detects_large_effect_eventually(self):
        rng = np.random.default_rng(0)
        test = SequentialTest(alpha=0.05)
        result = None
        for _ in range(50):
            c = rng.normal(0.0, 1.0, size=100)
            t = rng.normal(1.0, 1.0, size=100)
            result = test.update(c, t)
            if result.significant:
                break
        assert result is not None
        assert result.significant, "Large effect not detected after 5000 observations"

    def test_p_value_stays_valid_under_null(self):
        """False positive rate should stay near alpha under H0."""
        rng = np.random.default_rng(1)
        n_experiments = 500
        false_positives = 0
        for _ in range(n_experiments):
            test = SequentialTest(alpha=0.05)
            # One large batch under null
            c = rng.normal(0.0, 1.0, size=200)
            t = rng.normal(0.0, 1.0, size=200)
            result = test.update(c, t)
            if result.significant:
                false_positives += 1
        fpr = false_positives / n_experiments
        # Allow generous margin; the key property is it doesn't explode
        assert fpr < 0.15, f"False positive rate {fpr:.2%} too high"

    def test_p_value_between_zero_and_one(self):
        rng = np.random.default_rng(2)
        test = SequentialTest()
        result = test.update(rng.normal(size=100), rng.normal(size=100))
        assert 0.0 <= result.p_value <= 1.0

    def test_returns_sequential_result(self):
        rng = np.random.default_rng(3)
        test = SequentialTest()
        result = test.update(rng.normal(size=50), rng.normal(size=50))
        assert isinstance(result, SequentialResult)

    def test_p_value_one_for_tiny_samples(self):
        test = SequentialTest()
        result = test.update(np.array([1.0]), np.array([2.0]))
        assert result.p_value == 1.0

    def test_reset_clears_state(self):
        rng = np.random.default_rng(4)
        test = SequentialTest()
        test.update(rng.normal(size=200), rng.normal(size=200))
        test.reset()
        assert test._n_c == 0
        assert test._n_t == 0

    def test_incremental_updates_equivalent_to_batch(self):
        rng = np.random.default_rng(5)
        c_all = rng.normal(0.0, 1.0, size=300)
        t_all = rng.normal(0.5, 1.0, size=300)

        # Batch
        test_batch = SequentialTest()
        batch_result = test_batch.update(c_all, t_all)

        # Incremental (10 batches of 30)
        test_inc = SequentialTest()
        for i in range(10):
            inc_result = test_inc.update(c_all[i * 30:(i + 1) * 30], t_all[i * 30:(i + 1) * 30])

        assert batch_result.p_value == pytest.approx(inc_result.p_value, rel=1e-6)
        assert batch_result.effect_size == pytest.approx(inc_result.effect_size, rel=1e-6)

    def test_effect_size_tracks_mean_difference(self):
        rng = np.random.default_rng(6)
        c = rng.normal(5.0, 1.0, size=1000)
        t = rng.normal(6.0, 1.0, size=1000)
        test = SequentialTest()
        result = test.update(c, t)
        assert result.effect_size == pytest.approx(1.0, abs=0.15)
