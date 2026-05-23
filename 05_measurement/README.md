# Measurement

Comprehensive measurement toolkit covering experimental design, causal inference, and marketing attribution.

## Structure

```
05_measurement/
├── ab_testing/          A/B testing: z-test, t-test, CUPED, mSPRT sequential testing
├── causal_inference/    Causal methods: DiD, PSM, Bandits, Synthetic Control, MMM
├── experiment_analysis.ipynb
├── causal_analysis.ipynb
└── mmm_analysis.ipynb
```

## Quick start

```bash
python data/synthetic/generate_synthetic_data.py
jupyter notebook 05_measurement/experiment_analysis.ipynb
jupyter notebook 05_measurement/causal_analysis.ipynb
jupyter notebook 05_measurement/mmm_analysis.ipynb
```

## When to use which method

| Situation | Method |
|-----------|--------|
| Randomised experiment (A/B test) | `ab_testing` |
| Can't randomise; have pre/post and control group | DiD |
| Can't randomise; have rich covariates | PSM |
| High-traffic, want to minimise lost conversions during test | Multi-arm Bandit |
| Single treated unit with many donors (market, store) | Synthetic Control |
| Attribute revenue to marketing channels | MMM |

## A/B Testing (`ab_testing/`)

See `docs/modules/ab_testing.md` for the full guide.

```python
import sys; sys.path.insert(0, "05_measurement")
from ab_testing import Experiment, required_sample_size
from ab_testing.stats import z_test, cuped, wilson_ci
from ab_testing.sequential import SequentialTest
```

## Causal Inference (`causal_inference/`)

### Difference-in-Differences

```python
from causal_inference.did import DifferenceInDifferences

did = DifferenceInDifferences()
result = did.estimate_2x2(pre_treated, post_treated, pre_control, post_control)
print(result)  # ATT with CI and p-value

# Panel DiD (multiple units + time periods)
result = did.estimate_panel(df, "store_id", "week", "sales", "is_treated", treatment_week)

# Event study (visual parallel trends test)
es = did.event_study(df, "store_id", "week", "sales", "is_treated", treatment_week)
```

### Propensity Score Matching

```python
from causal_inference.psm import PropensityScoreMatching

psm = PropensityScoreMatching(caliper=0.05)
result = psm.match(df, treatment_col="is_treated", outcome_col="revenue",
                   covariate_cols=["age", "tenure", "plan"])
print(result)
print(result.balance)  # SMD before/after matching — check balance!
```

### Multi-arm Bandits

```python
from causal_inference.bandit import ThompsonSampling, EpsilonGreedy, UCB1

# Thompson Sampling (recommended for binary outcomes)
bandit = ThompsonSampling(n_arms=3, arm_names=["control", "variant_a", "variant_b"])
arm = bandit.select_arm()         # which variant to show next
bandit.update(arm, reward=1.0)    # record the outcome
print(bandit.summary())
```

### Synthetic Control

```python
from causal_inference.synthetic_control import SyntheticControl

sc = SyntheticControl()
result = sc.fit(df, treated_unit="store_A", treatment_time=10)
print(result)  # ATT + donor weights
placebo = sc.placebo_test(df, "store_A", treatment_time=10)
```

### Marketing Mix Modeling

```python
from causal_inference.mmm import MarketingMixModel, ChannelConfig

mmm = MarketingMixModel(
    channel_configs=[
        ChannelConfig("paid_search", decay=0.3, alpha=2.0, K=1000),
        ChannelConfig("social",      decay=0.6, alpha=1.5, K=500),
        ChannelConfig("email",       decay=0.1, alpha=3.0, K=150),
    ],
)
result = mmm.fit(df, outcome_col="revenue")
print(result)                              # R² + channel ROI table
budget_plan = mmm.optimize_budget(50_000) # optimal spend allocation
```
