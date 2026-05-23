# Looker Studio Dashboard Guide

## Prerequisites

- Looker Studio account (free at lookerstudio.google.com)
- BigQuery dataset with dbt marts populated
- IAM role: `BigQuery Data Viewer` on the analytics dataset

## Dashboard 1: Growth Overview

### Data source
`mart_mrr` + `dim_users`

### Charts

**1. MRR Waterfall (bar chart)**
- Dimension: `billing_month`
- Metrics: Sum of `amount_usd` broken by `mrr_type`
- Sort: `billing_month` ascending
- Color by: `mrr_type` (new=green, expansion=teal, contraction=orange, churned=red, renewal=grey)

**2. Cumulative MRR (line chart)**
- Calculated field: `RUNNING_SUM(SUM(amount_usd) FILTER mrr_type != 'churned_mrr')`
- Dimension: `billing_month`

**3. Signups by Channel (stacked bar)**
- Data source: `dim_users`
- Dimension: `DATE_TRUNC(signed_up_at, MONTH)`, `acquisition_channel`
- Metric: `COUNT(user_id)`

---

## Dashboard 2: Retention Heatmap

### Data source
`mart_cohort_retention`

### Charts

**1. Cohort Heatmap (pivot table)**
- Rows: `cohort_week`
- Columns: `period_number` (0–12)
- Metric: `AVG(retention_rate)`
- Conditional formatting: 0% = red, 100% = green

**2. Retention Curves by Cohort (line chart)**
- Dimension: `period_number`, `cohort_week`
- Metric: `AVG(retention_rate)`
- Filter: `period_number <= 12`

---

## Dashboard 3: Funnel Analysis

### Data source
`fct_funnel`

### Charts

**1. Funnel Bar Chart**
- Dimension: `step_name` (ordered by `step_order`)
- Metric: `users_reached`
- Filter: `funnel_name = 'activation_funnel'`

**2. Conversion Rate by Channel (cross-tab)**
- Join `fct_funnel` with `dim_users` on event counts
- Group by `acquisition_channel`

---

## Sharing

1. Share the report via **File → Share → Get report link**
2. Set permissions to "Anyone with the link can view" for stakeholder distribution
3. For scheduled email delivery: **File → Schedule email delivery**
