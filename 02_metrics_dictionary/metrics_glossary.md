# Metrics Glossary

Canonical definitions for every metric used in dashboards, experiments, and analyses.
**If a number is in a dashboard, its definition must be traceable to a row in this file.**

---

## Growth Metrics

| Metric | Definition | Unit | dbt Model |
|--------|-----------|------|-----------|
| **Signups** | Count of new `user_id` rows with a `signup_at` in the period | count | `dim_users` |
| **Conversion Rate** | Signups where `is_converted = true` / total signups in same cohort | % | `dim_users` |
| **Activation Rate** | Users who completed the activation funnel / signups in period | % | `fct_funnel` |
| **CAC** | Total acquisition spend / new converted users in period | USD | external |

## Retention Metrics

| Metric | Definition | Unit | dbt Model |
|--------|-----------|------|-----------|
| **Week-N Retention** | Users active in week N / cohort size at week 0 | % | `mart_cohort_retention` |
| **D30 Retention** | Users active 30 days after signup / cohort size | % | `mart_cohort_retention` |
| **Churn Rate** | Users churned in month / users active at start of month | % | `mart_mrr` |
| **Survival Rate** | 1 − cumulative churn | % | `mart_cohort_retention` |

## Revenue Metrics

| Metric | Definition | Unit | dbt Model |
|--------|-----------|------|-----------|
| **MRR** | Sum of `amount_usd` for all active subscriptions in the month | USD | `mart_mrr` |
| **New MRR** | MRR from users making their first payment | USD | `mart_mrr` |
| **Expansion MRR** | MRR from users upgrading their plan | USD | `mart_mrr` |
| **Contraction MRR** | MRR lost from users downgrading (negative) | USD | `mart_mrr` |
| **Churned MRR** | MRR lost from cancellations (negative) | USD | `mart_mrr` |
| **Net New MRR** | New + Expansion + Contraction + Churned | USD | `mart_mrr` |
| **ARPU** | MRR / paying users | USD | `mart_mrr` |
| **LTV** | ARPU / Monthly Churn Rate | USD | derived |

## Engagement Metrics

| Metric | Definition | Unit | dbt Model |
|--------|-----------|------|-----------|
| **DAU / WAU / MAU** | Distinct users with ≥1 non-page-view event in day/week/month | count | `int_event_counts` |
| **DAU/MAU ratio** | Stickiness proxy: DAU / MAU | ratio | derived |
| **Feature Adoption** | Users who used feature X / active users in period | % | `fct_funnel` |

---

## Calculation Notes

### MRR
We use the **recognition** approach: MRR is the monthly recurring charge, not cash received.
- Annual plans are divided by 12 and recognised monthly.
- Free users contribute $0 MRR.
- Churn MRR is negative and represents the last period's MRR from churned users.

### Cohort Retention
- **Cohort** = calendar week of first event (not just signup — use `dim_users.signed_up_at`).
- **Active** = at least 1 non-page-view event in the week.
- Period 0 = signup week, always 100%.

### Conversion Rate
Denominator is **all signups**, not just those given enough time to convert.
For time-adjusted rates, filter to cohorts ≥ 30 days old.

---

## Experiment Metrics

| Metric | Type | Test | Notes |
|--------|------|------|-------|
| Conversion rate | Binary | z-test | Use Wilson CI for reporting |
| Revenue per user | Continuous (skewed) | Mann-Whitney or CUPED | Log-transform before t-test if log-normal |
| Feature adoption rate | Binary | z-test | |
| Session duration | Continuous (skewed) | Mann-Whitney | |
| Retention at D7 | Binary | z-test | Requires waiting 7 days post-exposure |
