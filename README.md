# data-bedrock

![data-bedrock](assets/banner.png)

A production-ready, opinionated analytics toolkit for the first Data Scientist at a seed/Series A startup.

## What's inside

| Module | Purpose |
|--------|---------|
| `00_tracking_plan/` | Event taxonomy and Segment schema |
| `01_dbt_project/` | dbt models: staging → intermediate → marts |
| `02_metrics_dictionary/` | Metric definitions and dbt metrics YAML |
| `03_growth_analysis/` | Funnel and acquisition analysis |
| `04_cohort_retention/` | Cohort heatmaps and retention curves |
| `05_measurement/` | A/B testing, causal inference (DiD, PSM, Bandits, Synthetic Control, MMM) |
| `06_dashboards/` | Metabase exports and Looker Studio guide |
| `data/synthetic/` | Synthetic SaaS event data generator |
| `docs/` | Architecture, module guides, decision logs |

## Quick start

```bash
# 1. Set up Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Copy env template
cp .env.example .env

# 3. Generate synthetic data (required before running notebooks)
python data/synthetic/generate_synthetic_data.py

# 4. Run tests
pytest tests/ -v

# 5. dbt (from 01_dbt_project/)
cd 01_dbt_project
cp profiles.yml.example profiles.yml  # edit with your credentials
dbt debug
dbt run
dbt test
```

## Design principles

- **Synthetic data first** — every module runs against `data/synthetic/` with no warehouse needed
- **Library code lives in the module packages** — notebooks import from them, tests cover them
- **dbt is the single transformation layer** — Python is for analysis, not ETL
- **Stats assumptions are explicit** — all stat calls go through wrappers, never raw `scipy`

## Stack

Python 3.11+ · SQL (BigQuery) · dbt Core · Jupyter · Pandas · Scipy
