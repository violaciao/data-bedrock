# Growth Analysis

Funnel conversion analysis and acquisition channel analytics.

## Quick start

```bash
python data/synthetic/generate_synthetic_data.py
jupyter notebook 03_growth_analysis/funnel_analysis.ipynb
jupyter notebook 03_growth_analysis/acquisition_analysis.ipynb
```

## Library

```
src/
├── funnel.py      FunnelAnalysis class — ordered step-by-step conversion
└── acquisition.py acquisition_metrics, channel_comparison
```

## Key functions

- `FunnelAnalysis(steps=[...]).compute(events)` — returns per-step conversion rates and user counts.
- `acquisition_metrics(users, orders)` — signups, conversion rate, ARPU per channel.
- `channel_comparison(users)` — monthly time-series of signups by channel.
