# Module Guide: Cohort & Retention (`04_cohort_retention/`)

## Overview

Builds weekly cohort retention matrices, fits parametric retention curves, and
compares retention across acquisition channels.

## Quick start

```python
import sys
sys.path.insert(0, "04_cohort_retention")

from cohort_retention.cohort import build_cohort_matrix
from cohort_retention.retention import fit_retention_curve, retention_by_channel

# Build the retention matrix
matrix = build_cohort_matrix(events_df)
# matrix.index = cohort week strings ("2024-01-01/2024-01-07")
# matrix.columns = [0, 1, 2, ..., 25]  (weeks since first activity)
# matrix.values = retention rates (0–1), NaN for future periods

# Average retention curve across all cohorts
avg = matrix.mean(axis=0).dropna()

# Fit a power-law model
fit = fit_retention_curve(avg, model="power_law")
print(f"1-year estimated retention: {fit['long_run_estimate']:.1%}")
print(f"R² = {fit['r_squared']:.3f}")

# Retention split by channel
channel_ret = retention_by_channel(users_df, events_df)
```

## Function reference

### `build_cohort_matrix(events, user_col, date_col, period, max_periods)`

| Param | Default | Description |
|-------|---------|-------------|
| `events` | — | Event-level DataFrame |
| `user_col` | `"user_id"` | User identifier column |
| `date_col` | `"occurred_at"` | Timestamp column |
| `period` | `"W"` | `"W"` weekly, `"M"` monthly |
| `max_periods` | `26` | Number of periods to include |

Returns a DataFrame where:
- **Rows** = cohort periods (first-activity week/month)
- **Columns** = integer period offsets (0, 1, 2 …)
- **Values** = retention rate (active users / cohort size), NaN for periods not yet observed

Period 0 is always 1.0 by definition.

### `fit_retention_curve(series, model)`

Fits a parametric model to the average retention curve.

| Model | Formula | Best for |
|-------|---------|----------|
| `power_law` | `r(t) = a · (t+1)^(-b)` | Most SaaS products — retention stabilises |
| `exponential` | `r(t) = a · e^(-bt)` | Products with constant churn rate |

Returns a dict with `params`, `fitted`, `r_squared`, `long_run_estimate` (at t=52).

### `retention_by_channel(users, events, periods)`

Returns a DataFrame with one row per acquisition channel and columns `[channel, 0, 1, ..., periods-1]`
showing average retention at each period offset.

## Interpreting results

- **Period-1 retention** (week after signup) is the single most important number — it predicts
  long-run retention better than any other metric.
- A **power-law curve** that flattens above 10% suggests a healthy engaged user base.
- **Channel comparison** often reveals that referral users have significantly higher retention
  than paid acquisition — use this to inform CAC/LTV calculations.

## Notebooks

| Notebook | What it shows |
|----------|--------------|
| `cohort_heatmap.ipynb` | Full retention matrix as a colour-coded heatmap + average curve |
| `retention_curves.ipynb` | Power-law fit + per-channel retention curves |
