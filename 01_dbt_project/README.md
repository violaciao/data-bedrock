# dbt Project

BigQuery-first dbt project. Staging → Intermediate → Marts.

## Setup

```bash
cd 01_dbt_project
cp profiles.yml.example profiles.yml  # edit with your GCP credentials
dbt deps
dbt debug
```

## Run

```bash
dbt run --select staging          # views in analytics_dev.staging
dbt run --select marts            # tables in analytics_dev.marts
dbt test                          # schema + data tests
dbt docs generate && dbt docs serve
```

## Models

| Model | Layer | Description |
|-------|-------|-------------|
| `stg_users` | staging | Cleaned users |
| `stg_events` | staging | Cleaned event stream |
| `stg_orders` | staging | Cleaned orders |
| `int_user_spine` | intermediate | User × calendar week spine |
| `int_event_counts` | intermediate | Weekly event counts per user |
| `dim_users` | mart | User dimension with activity rollups |
| `fct_funnel` | mart | Funnel conversion (parameterized by seed) |
| `mart_cohort_retention` | mart | Cohort × period retention matrix |
| `mart_mrr` | mart | Monthly MRR by type |

## Adding a new funnel

Edit `seeds/funnel_config.csv`, then run:

```bash
dbt seed
dbt run --select fct_funnel
```
