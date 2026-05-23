# Module Guide: A/B Testing (`05_measurement/ab_testing/`)

## Overview

The A/B testing module provides a complete experiment infrastructure:
- Statistical tests with explicit assumptions
- Sample size / power analysis
- Sequential (always-valid) testing for early stopping
- High-level `Experiment` class for end-to-end workflows

## When to use each test

```
Metric type?
├── Binary (converted / not)
│   └── z_test()
└── Continuous
    ├── Have pre-experiment data?
    │   └── cuped()          ← best power
    ├── Skewed metric (|skewness| > 2)?
    │   └── mann_whitney_u_test()
    └── Otherwise
        └── welch_t_test()
```

## Quick start

```python
import sys
sys.path.insert(0, "05_measurement")
from ab_testing import Experiment, required_sample_size

# 1. Plan the experiment
result = required_sample_size(
    baseline_rate=0.05,
    mde_relative=0.20,     # detect a 20% lift (5% → 6%)
    alpha=0.05,
    power=0.80,
    daily_traffic_per_variant=500,
)
print(result)
# → need 3,843 per variant, ~8 days at 500/day

# 2. Analyse after the experiment
exp = Experiment("checkout_button_color", metric_type="binary")
stat_result = exp.analyze(control_converted, treatment_converted)
print(stat_result)
```

## Always-valid / sequential testing

Use `SequentialTest` when you want to peek at results before the planned end date
without inflating false positives:

```python
from ab_testing.sequential import SequentialTest

test = SequentialTest(alpha=0.05)
for daily_batch in streaming_data:
    result = test.update(daily_batch["control"], daily_batch["treatment"])
    if result.significant:
        print("Stop the experiment now:", result)
        break
```

The p-value from `SequentialTest` is valid at every observation — you can stop
at any time without correction. This is not true of standard tests.

## CUPED

CUPED reduces the noise in your metric by removing variance explained by a
pre-experiment measurement:

```python
from ab_testing.stats import cuped

result = cuped(
    control_post=revenue_last_week_control,
    treatment_post=revenue_last_week_treatment,
    control_pre=revenue_prior_week_control,
    treatment_pre=revenue_prior_week_treatment,
)
```

Typical variance reduction: 20–50% if the pre/post correlation is > 0.5.
This is equivalent to running the experiment on a ~30–100% larger sample.

## Reporting

Always report:
1. Effect size with confidence interval (not just p-value)
2. The test used and why
3. Sample sizes in both variants
4. Whether the experiment was pre-registered or peeked at early

See `experiment_analysis.ipynb` for a full worked example.

The measurement area also includes separate walkthrough notebooks for
Difference-in-Differences, Propensity Score Matching, Synthetic Control, and
Marketing Mix Modeling in `05_measurement/`.
