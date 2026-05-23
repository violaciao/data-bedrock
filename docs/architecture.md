# Architecture

## Data flow

```
Raw events (Segment / SDK)
        │
        ▼
  Ingestion layer
  (Fivetran / Airbyte → BigQuery raw schema)
        │
        ▼
  dbt transformation
  ┌──────────────────────────────────┐
  │  staging/   light cleaning      │
  │  intermediate/  date spines     │
  │  marts/     business entities   │
  └──────────────────────────────────┘
        │
        ▼
  BI / Analysis
  ┌────────────┬────────────┬────────────────┐
  │ Metabase   │ Looker     │ Jupyter        │
  │ dashboards │ Studio     │ notebooks      │
  └────────────┴────────────┴────────────────┘
```

## Layer responsibilities

### Staging (`stg_`)
- One model per source table
- Light renaming and casting only
- No joins, no business logic
- Materialised as views (cheap, always fresh)

### Intermediate (`int_`)
- Joins between staging models
- Date spines (always here, never in marts)
- Reusable building blocks for multiple marts
- Materialised as ephemeral (compiled inline)

### Marts (`fct_`, `dim_`, `mart_`)
- Business-facing tables
- `dim_` = slowly changing dimensions (users, accounts)
- `fct_` = event-grain facts (funnel events, activity)
- `mart_` = pre-aggregated reporting tables (MRR, retention)
- Materialised as tables (fast BI queries)

## Module boundaries

Each analysis module (`03_growth_analysis/`, `04_cohort_retention/`, `05_measurement/`) is self-contained:

```
module/
├── README.md          instructions
├── <package>/         reusable Python library
│   ├── __init__.py
│   └── *.py
└── *.ipynb            presentation notebooks (import from the package)
```

Examples:
- `03_growth_analysis/growth_analysis/`
- `04_cohort_retention/cohort_retention/`
- `05_measurement/ab_testing/`
- `05_measurement/causal_inference/`

**Rule:** Notebooks import from the module package. Package code never imports from notebooks.
**Rule:** No cross-module imports (e.g. `04_cohort_retention` must not import from `03_growth_analysis`).

## Testing strategy

| Layer | Test type | Where |
|-------|-----------|-------|
| dbt models | schema tests (not_null, unique, accepted_values) | `*.yml` |
| dbt models | data tests (custom SQL assertions) | `tests/` in dbt project |
| Python packages | unit tests | `tests/` at repo root |
| Notebooks | smoke test: Run All against synthetic data | manual / CI |

All Python tests must run **offline** against `data/synthetic/` — never against a live warehouse.

## Environment variables

See `.env.example` for all required variables. The `.env` file is gitignored.

dbt connection credentials live in `01_dbt_project/profiles.yml` (also gitignored).
