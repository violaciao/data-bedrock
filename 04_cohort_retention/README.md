# Cohort & Retention Analysis

Weekly cohort retention heatmaps, survival curves, and parametric retention curve fitting.

## Quick start

```bash
python data/synthetic/generate_synthetic_data.py
jupyter notebook 04_cohort_retention/cohort_heatmap.ipynb
jupyter notebook 04_cohort_retention/retention_curves.ipynb
```

## Library

```
src/
├── cohort.py     build_cohort_matrix — cohort × period retention matrix
└── retention.py  fit_retention_curve, retention_by_channel
```

## Key functions

- `build_cohort_matrix(events)` — returns a DataFrame where rows are cohort periods and columns are period offsets (0, 1, 2 …), values are retention rates.
- `fit_retention_curve(series, model='power_law')` — fits a power-law or exponential model to the average retention curve.
- `retention_by_channel(users, events)` — splits the cohort matrix by acquisition channel.
