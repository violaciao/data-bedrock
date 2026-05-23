"""Causal inference toolkit.

Methods:
- Difference-in-Differences (did.py)
- Propensity Score Matching (psm.py)
- Multi-arm Bandits (bandit.py)
- Marketing Mix Modeling (mmm.py)
- Synthetic Control (synthetic_control.py)
"""

from .bandit import EpsilonGreedy, ThompsonSampling, UCB1
from .did import DifferenceInDifferences
from .mmm import MarketingMixModel
from .psm import PropensityScoreMatching
from .synthetic_control import SyntheticControl

__all__ = [
    "DifferenceInDifferences",
    "PropensityScoreMatching",
    "EpsilonGreedy",
    "ThompsonSampling",
    "UCB1",
    "MarketingMixModel",
    "SyntheticControl",
]
