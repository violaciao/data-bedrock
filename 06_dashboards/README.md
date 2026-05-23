# Dashboards

## Metabase

`metabase/dashboard_export.json` contains a Metabase dashboard export with:
- MRR waterfall (new / expansion / contraction / churn)
- Weekly cohort retention heatmap
- Acquisition channel funnel comparison
- DAU/WAU/MAU trends

### Import

1. In Metabase: **Settings → Admin → Import**
2. Upload `metabase/dashboard_export.json`
3. Remap the database connection to your BigQuery instance

### Requirements
- Metabase 0.47+
- `mart_mrr`, `mart_cohort_retention`, `fct_funnel`, `dim_users` tables populated

## Looker Studio

See `looker_studio_guide.md` for step-by-step instructions to recreate the
standard dashboard suite in Looker Studio (free, no self-hosting required).

Recommended for teams that are already on GCP and want zero infrastructure.
