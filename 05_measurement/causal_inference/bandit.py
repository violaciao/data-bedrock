"""Multi-arm Bandit algorithms for adaptive experimentation.

Traditional A/B tests fix allocation (50/50) for the full duration and only
read results at the end. Bandits continuously update allocation to favour
better-performing variants, reducing regret (lost conversions during the test).

The tradeoff: bandits converge faster to the winner but lose some statistical
rigour — they are not designed to produce frequentist p-values. Use them when:
- Minimising regret during the experiment matters (e.g. high-traffic revenue tests)
- You care more about finding the winner than about a precise effect estimate
- You are willing to accept slight bias in the final estimates

For rigorous causal inference, use the A/B testing module instead.

Algorithms provided:
- **Epsilon-Greedy**: exploit the current best arm with probability 1-ε, explore randomly otherwise.
- **Thompson Sampling**: Bayesian approach; samples from posterior Beta distributions.
  Natural choice for binary outcomes (conversion rates).
- **UCB1**: Upper Confidence Bound; deterministically balances exploration and exploitation
  by selecting the arm with the highest (mean + confidence bonus).

Design decisions:
- All algorithms use the same interface: ``update(arm, reward)`` + ``select_arm()``.
- Thompson Sampling uses Beta-Bernoulli conjugate priors (binary rewards only).
- UCB1 uses the standard log(t)/n_i confidence bonus.
- Regret tracking is built in; call ``regret_df()`` to get a time-series.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BanditBase(ABC):
    """Abstract base class for multi-arm bandit algorithms.

    Args:
        n_arms: Number of variants (arms).
        arm_names: Optional list of arm labels. Defaults to ``["arm_0", ...]``.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_arms: int,
        arm_names: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.n_arms = n_arms
        self.arm_names = arm_names or [f"arm_{i}" for i in range(n_arms)]
        self.rng = np.random.default_rng(seed)
        self._counts = np.zeros(n_arms, dtype=int)
        self._totals = np.zeros(n_arms, dtype=float)
        self._history: list[dict] = []

    @abstractmethod
    def select_arm(self) -> int:
        """Select the next arm to pull. Returns arm index (0-based)."""

    def update(self, arm: int, reward: float) -> None:
        """Record the reward for a pulled arm.

        Args:
            arm: Index of the arm that was pulled.
            reward: Observed reward (0/1 for binary, any float for continuous).
        """
        self._counts[arm] += 1
        self._totals[arm] += reward
        self._history.append({"t": len(self._history) + 1, "arm": arm, "reward": reward})

    @property
    def means(self) -> np.ndarray:
        """Current mean reward estimate per arm."""
        with np.errstate(invalid="ignore"):
            m = self._totals / np.where(self._counts > 0, self._counts, 1)
            m[self._counts == 0] = 0.0
        return m

    def best_arm(self) -> int:
        """Return the index of the arm with the highest current mean."""
        return int(np.argmax(self.means))

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame with current estimates for each arm."""
        return pd.DataFrame({
            "arm": self.arm_names,
            "pulls": self._counts,
            "total_reward": self._totals,
            "mean_reward": self.means,
        })

    def regret_df(self, true_best_mean: float) -> pd.DataFrame:
        """Compute cumulative regret over time.

        Args:
            true_best_mean: The true mean reward of the optimal arm.

        Returns:
            DataFrame with columns: t, arm, reward, instantaneous_regret,
            cumulative_regret.
        """
        df = pd.DataFrame(self._history)
        if df.empty:
            return df
        df["arm_mean_at_pull"] = df["arm"].map(dict(enumerate(self.means)))
        df["instantaneous_regret"] = true_best_mean - df["reward"]
        df["cumulative_regret"] = df["instantaneous_regret"].cumsum()
        return df


class EpsilonGreedy(BanditBase):
    """Epsilon-greedy bandit.

    Exploits the current best arm with probability (1 - epsilon), and
    explores a random arm with probability epsilon.

    Args:
        n_arms: Number of arms.
        epsilon: Exploration rate in [0, 1]. Use ``epsilon=1.0`` for pure
            exploration (uniform random), ``epsilon=0.0`` for pure greedy.
        decay: Multiplicative decay applied to epsilon each round. Set to
            ``1.0`` (no decay) for a fixed epsilon.
        arm_names: Optional arm labels.
        seed: Random seed.
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.10,
        decay: float = 1.0,
        arm_names: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(n_arms, arm_names, seed)
        self.epsilon = epsilon
        self.decay = decay

    def select_arm(self) -> int:
        if self.rng.random() < self.epsilon:
            arm = int(self.rng.integers(0, self.n_arms))
        else:
            arm = self.best_arm()
        self.epsilon *= self.decay
        return arm


class ThompsonSampling(BanditBase):
    """Thompson Sampling with Beta-Bernoulli conjugate prior.

    Best for binary reward signals (conversion rate, click-through rate).
    Each arm has a Beta(α, β) posterior over its conversion rate.
    At each step, we sample from each posterior and pull the arm with the
    highest sample.

    Args:
        n_arms: Number of arms.
        prior_alpha: Prior successes (default 1 = uniform prior).
        prior_beta: Prior failures (default 1 = uniform prior).
        arm_names: Optional arm labels.
        seed: Random seed.
    """

    def __init__(
        self,
        n_arms: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        arm_names: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(n_arms, arm_names, seed)
        self._alpha = np.full(n_arms, prior_alpha)
        self._beta = np.full(n_arms, prior_beta)

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        self._alpha[arm] += float(reward)
        self._beta[arm] += float(1 - reward)

    def select_arm(self) -> int:
        samples = self.rng.beta(self._alpha, self._beta)
        return int(np.argmax(samples))

    def posterior_summary(self) -> pd.DataFrame:
        """Return posterior Beta distribution parameters per arm."""
        return pd.DataFrame({
            "arm": self.arm_names,
            "alpha": self._alpha,
            "beta": self._beta,
            "posterior_mean": self._alpha / (self._alpha + self._beta),
            "posterior_std": np.sqrt(
                self._alpha * self._beta
                / ((self._alpha + self._beta) ** 2 * (self._alpha + self._beta + 1))
            ),
        })


class UCB1(BanditBase):
    """Upper Confidence Bound (UCB1) bandit.

    Selects the arm with the highest upper confidence bound:
        UCB(i) = x̄_i + √(2 · log(t) / n_i)

    This is deterministic given the history — no randomness after initialisation.
    Each arm is pulled once before UCB kicks in.

    Args:
        n_arms: Number of arms.
        arm_names: Optional arm labels.
        seed: Random seed (only used for tie-breaking in the first n_arms rounds).
    """

    def select_arm(self) -> int:
        t = len(self._history) + 1
        # Pull each arm once before computing UCB
        unpulled = np.where(self._counts == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])
        ucb_values = self.means + np.sqrt(2 * np.log(t) / self._counts)
        return int(np.argmax(ucb_values))
