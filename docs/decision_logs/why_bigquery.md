# Decision: BigQuery as the default warehouse

**Date:** January 2024
**Status:** Accepted

## Context

We needed to choose a default warehouse for the dbt project. The main candidates were BigQuery, Snowflake, and Redshift.

## Decision

Default to **BigQuery** with Snowflake notes in comments where syntax differs significantly.

## Rationale

1. **Cost model** — BigQuery's on-demand pricing (pay per query) has zero fixed cost, which is ideal for a seed startup that isn't running queries 24/7. Snowflake's per-second credit model becomes cheaper at higher utilisation, but that's a later problem.

2. **GCP ecosystem** — Startups using Google Workspace or Firebase already have GCP accounts, lowering the barrier to a first warehouse.

3. **BigQuery ML** — Native ML functions are useful if the team wants to do basic propensity scoring without standing up a separate ML platform.

4. **Standard SQL** — BigQuery uses ANSI SQL. Most dbt syntax is portable; we note Snowflake/Redshift divergences inline.

## Tradeoffs

- **BigQuery** doesn't have a local emulator, making dbt development harder without a real GCP project. We mitigate this by running Python tests against synthetic CSVs, not dbt.
- **Snowflake** has better time-travel and data-sharing features, which matter more at Series B+.
- **Redshift** is the right choice if the company is all-in on AWS.

## Migration path

If the company switches warehouses, the primary changes are:
1. `profiles.yml` adapter
2. A handful of BigQuery-specific functions (`date_trunc` → `date_trunc` in Snowflake is the same; `countif` → `count(case when ...)`)
3. `dbt-bigquery` → `dbt-snowflake` in `requirements.txt`
