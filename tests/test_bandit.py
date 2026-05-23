"""Tests for 05_measurement/causal_inference/bandit.py."""

from __future__ import annotations

import numpy as np
import pytest

from causal_inference.bandit import EpsilonGreedy, ThompsonSampling, UCB1


TRUE_RATES = [0.05, 0.10, 0.15]  # arm 2 is best
N_ROUNDS = 3000


def _run_bandit(bandit, true_rates, n_rounds, seed=42):
    rng = np.random.default_rng(seed)
    for _ in range(n_rounds):
        arm = bandit.select_arm()
        reward = float(rng.random() < true_rates[arm])
        bandit.update(arm, reward)


class TestEpsilonGreedy:
    def test_converges_to_best_arm(self):
        # ε-greedy converges slower than Thompson/UCB1; use more rounds
        b = EpsilonGreedy(3, epsilon=0.1, seed=0)
        _run_bandit(b, TRUE_RATES, 6000)
        assert b.best_arm() == 2

    def test_pulls_are_recorded(self):
        b = EpsilonGreedy(3, epsilon=0.1, seed=0)
        _run_bandit(b, TRUE_RATES, 100)
        assert b._counts.sum() == 100

    def test_decay_reduces_epsilon(self):
        b = EpsilonGreedy(3, epsilon=0.5, decay=0.99, seed=0)
        initial_eps = b.epsilon
        b.select_arm()
        assert b.epsilon < initial_eps

    def test_summary_shape(self):
        b = EpsilonGreedy(3, seed=0)
        _run_bandit(b, TRUE_RATES, 100)
        df = b.summary()
        assert df.shape == (3, 4)

    def test_regret_df(self):
        b = EpsilonGreedy(3, epsilon=0.1, seed=0)
        _run_bandit(b, TRUE_RATES, 100)
        df = b.regret_df(true_best_mean=TRUE_RATES[-1])
        assert "cumulative_regret" in df.columns
        assert df["cumulative_regret"].iloc[-1] >= 0

    def test_pure_exploit_always_picks_best(self):
        b = EpsilonGreedy(3, epsilon=0.0, seed=0)
        # Use deterministic rewards so arm 2 is unambiguously the best
        for _ in range(10):
            b.update(0, 0.0)   # arm 0 mean → 0.00
            b.update(1, 0.5)   # arm 1 mean → 0.50
            b.update(2, 1.0)   # arm 2 mean → 1.00
        for _ in range(20):
            assert b.select_arm() == 2


class TestThompsonSampling:
    def test_converges_to_best_arm(self):
        b = ThompsonSampling(3, seed=0)
        _run_bandit(b, TRUE_RATES, N_ROUNDS)
        assert b.best_arm() == 2

    def test_posterior_summary(self):
        b = ThompsonSampling(3, seed=0)
        _run_bandit(b, TRUE_RATES, 500)
        ps = b.posterior_summary()
        assert set(ps.columns) >= {"arm", "alpha", "beta", "posterior_mean"}

    def test_posterior_mean_tracks_true_rate(self):
        b = ThompsonSampling(3, seed=0)
        _run_bandit(b, TRUE_RATES, N_ROUNDS)
        ps = b.posterior_summary()
        for i, rate in enumerate(TRUE_RATES):
            assert ps.loc[i, "posterior_mean"] == pytest.approx(rate, abs=0.05)

    def test_prior_alpha_beta_influence(self):
        # Strong prior toward 0.5 should slow convergence
        b = ThompsonSampling(2, prior_alpha=100.0, prior_beta=100.0, seed=0)
        b.update(0, 1.0)
        b.update(1, 0.0)
        ps = b.posterior_summary()
        # Posterior means should be pulled toward 0.5 despite one observation each
        assert abs(ps.loc[0, "posterior_mean"] - 0.5) < 0.1
        assert abs(ps.loc[1, "posterior_mean"] - 0.5) < 0.1


class TestUCB1:
    def test_pulls_each_arm_once_first(self):
        b = UCB1(3, seed=0)
        # Must call update() after each select so _counts advances
        arms_pulled = []
        for _ in range(3):
            arm = b.select_arm()
            arms_pulled.append(arm)
            b.update(arm, 0.5)
        assert set(arms_pulled) == {0, 1, 2}

    def test_converges_to_best_arm(self):
        b = UCB1(3, seed=0)
        _run_bandit(b, TRUE_RATES, N_ROUNDS)
        assert b.best_arm() == 2

    def test_allocation_favours_best(self):
        b = UCB1(3, seed=0)
        _run_bandit(b, TRUE_RATES, N_ROUNDS)
        counts = b._counts
        assert counts[2] > counts[0]  # best arm pulled more than worst

    def test_deterministic_given_history(self):
        b = UCB1(3, seed=0)
        # After the same updates, select_arm must be deterministic
        for arm in range(3):
            b.update(arm, 0.5)
        arm1 = b.select_arm()
        arm2 = b.select_arm()
        assert arm1 == arm2  # no randomness in UCB1
