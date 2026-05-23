# data-bedrock

![data-bedrock](assets/banner.png)

Production-ready analytics starter kit for the first data scientist, analytics engineer, or growth lead at a startup.

`data-bedrock` combines product analytics, dbt models, metrics definitions, cohort retention, A/B testing, causal inference, marketing mix modeling, and dashboard templates in one repo you can actually fork and adapt.

## Why `data-bedrock` exists

Most early-stage teams do not need another toy notebook. They need a practical analytics foundation:

- A tracking plan that keeps events consistent
- A dbt project that turns raw data into trustworthy marts
- A metrics dictionary that stops KPI drift
- Analysis code for funnels, retention, experimentation, and growth
- Dashboard templates that shorten time to first decision

This repository is built for that exact job.

## `data-bedrock` is designed for:

- Startup data scientists building the first analytics system
- Analytics engineers setting up dbt models and KPI definitions
- Growth teams that need funnel, retention, and acquisition analysis
- Product teams that want an experimentation and measurement toolkit
- Portfolio companies or agencies that need a reusable analytics baseline

## What You Get

| Area | What is included |
|------|------------------|
| `00_tracking_plan/` | Event taxonomy, tracking plan, and Segment schema starter |
| `01_dbt_project/` | BigQuery-first dbt project with staging, intermediate, and mart layers |
| `02_metrics_dictionary/` | Business metric definitions and dbt-friendly metric specs |
| `03_growth_analysis/` | Funnel conversion and acquisition channel analysis |
| `04_cohort_retention/` | Cohort heatmaps, retention curves, and retention modeling |
| `05_measurement/` | A/B testing, sequential testing, DiD, PSM, bandits, synthetic control, and MMM |
| `06_dashboards/` | Metabase dashboard export and Looker Studio implementation guide |
| `data/synthetic/` | Synthetic SaaS data generator so the repo works before your warehouse is ready |
| `tests/` | Test coverage for core statistical and analytics modules |
| `docs/` | Architecture notes, module guides, and decision logs |

## Highlights

- It covers the full analytics workflow, from instrumentation to dashboards.
- It is opinionated enough to be practical, but modular enough to adapt piece by piece.
- It includes synthetic data, so you can explore the system before connecting production sources.
- It packages reusable code instead of burying logic only in notebooks.
- It offers early-stage teams a concrete reference for how modern analytics projects fit together.

## Core Use Cases

- Build a startup analytics repo from scratch in a weekend
- Stand up dbt models for product events, orders, users, MRR, funnels, and retention
- Create a shared source of truth for business metrics
- Run acquisition, conversion, and retention analysis on realistic sample data
- Evaluate experiments with standard and sequential testing methods
- Apply causal inference methods when randomized tests are not possible
- Launch baseline executive and growth dashboards faster

## Quick Start

```bash
# 1. Create a Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure local environment
cp .env.example .env

# 3. Generate synthetic data
python data/synthetic/generate_synthetic_data.py

# 4. Run tests
pytest tests/ -v

# 5. Run dbt models
cd 01_dbt_project
cp profiles.yml.example profiles.yml
dbt debug
dbt run
dbt test
```

## Tech Stack

Python 3.11+ · dbt Core · BigQuery · SQL · Jupyter · Pandas · NumPy · SciPy · Metabase · Looker Studio

## Project Design Principles

- Synthetic data first: the repo is usable before you connect your warehouse
- dbt as the transformation layer: business logic belongs in models, not scattered notebooks
- Reusable Python modules: analysis code is importable and testable
- Explicit statistical assumptions: experiments and inference methods are wrapped with consistent interfaces
- Startup pragmatism: prioritize artifacts teams can ship, review, and maintain

## Repository Map

```text
data-bedrock/
├── 00_tracking_plan/        tracking plan and event schema
├── 01_dbt_project/          dbt models for staging, marts, and analytics tables
├── 02_metrics_dictionary/   KPI definitions and metric glossary
├── 03_growth_analysis/      funnel and acquisition analysis
├── 04_cohort_retention/     retention analysis and curve fitting
├── 05_measurement/          experimentation and causal inference toolkit
├── 06_dashboards/           BI templates and dashboard setup guides
├── data/synthetic/          synthetic SaaS dataset generator
├── docs/                    architecture and module documentation
└── tests/                   automated test suite
```

## Notable Assets

- `fct_funnel` dbt model with configurable funnel steps
- `mart_cohort_retention` and `mart_mrr` analytics marts
- `FunnelAnalysis` utilities for ordered conversion analysis
- Cohort matrix and retention curve fitting helpers
- Sequential testing and sample size utilities for experiments
- Causal inference modules for DiD, PSM, bandits, synthetic control, and MMM
- Metabase export with MRR, retention, funnel, and engagement dashboards

## Best Way To Adapt It

1. Fork the repo.
2. Replace the synthetic data inputs with your warehouse sources.
3. Customize the tracking plan and metric definitions for your product.
4. Swap in your dbt profile and warehouse credentials.
5. Keep the modules you need and delete the rest.

## Search Keywords

If you found this repo while looking for a startup analytics template, these are the problems it is meant to solve: product analytics, analytics engineering, dbt starter project, cohort retention analysis, funnel analysis, A/B testing framework, causal inference in Python, marketing mix modeling, growth analytics, metrics layer, and startup data stack.

## Related Docs

- [Playbook](docs/playbook.md)
- [Architecture](docs/architecture.md)
- [dbt project guide](01_dbt_project/README.md)
- [Growth analysis guide](03_growth_analysis/README.md)
- [Cohort retention guide](04_cohort_retention/README.md)
- [Measurement guide](05_measurement/README.md)
- [Dashboard guide](06_dashboards/README.md)

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
