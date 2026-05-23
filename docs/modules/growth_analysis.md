# Module Guide: Growth Analysis (`03_growth_analysis/`)

## Overview

Funnel conversion analysis and acquisition channel analytics. The two libraries
are independent — use them separately or together.

## Quick start

```python
import sys
sys.path.insert(0, "03_growth_analysis")

from growth_analysis.funnel import FunnelAnalysis
from growth_analysis.acquisition import acquisition_metrics, channel_comparison

# --- Funnel ---
funnel = FunnelAnalysis(
    steps=["page_view", "signup", "feature_used", "upgrade_clicked"]
)
results = funnel.compute(events_df)
df = FunnelAnalysis.to_dataframe(results)
print(df[["event", "users", "conversion_from_top", "conversion_from_previous"]])

# --- Acquisition ---
metrics = acquisition_metrics(users_df, orders_df)
# → signups, conversions, conversion_rate, total_revenue, arpu per channel
```

## Function reference

### `FunnelAnalysis`

```python
FunnelAnalysis(
    steps: list[str],           # Ordered event types defining the funnel
    user_col: str = "user_id",
    event_col: str = "event_type",
    date_col: str | None = "occurred_at",  # Optional; used for ordering if present
)
```

**`.compute(events)`** — returns `list[FunnelStep]` with:

| Field | Description |
|-------|-------------|
| `users` | Distinct users who reached this step (having completed all prior steps) |
| `conversion_from_top` | `users / top_of_funnel_users` |
| `conversion_from_previous` | `users / previous_step_users` |
| `dropped` | Users lost between previous step and this step |

**`.to_dataframe(results)`** — converts the list to a tidy DataFrame.

### `acquisition_metrics(users, orders=None)`

Per-channel summary: signups, conversions, conversion rate, and (if orders provided)
total revenue and ARPU.

### `channel_comparison(users, period="M")`

Monthly time-series of signups and conversions by channel. Feed this into a line
chart to track channel mix over time.

## Design notes

- **No time window is applied automatically.** Filter your events DataFrame before
  passing it in. The notebooks apply a `LOOKBACK_DAYS` filter.
- **Users must complete steps in order** to count at a given step. This is the
  standard "ordered funnel" semantics, not "ever did the event" semantics.
- **New funnels** in dbt are added via `seeds/funnel_config.csv` — no SQL changes
  required. The Python `FunnelAnalysis` class is for ad-hoc notebook analysis.

## Notebooks

| Notebook | What it shows |
|----------|--------------|
| `funnel_analysis.ipynb` | Activation and conversion funnels with step-over-step bars |
| `acquisition_analysis.ipynb` | Channel conversion rates, ARPU, and monthly signup trends |
