# data-bedrock Playbook

A step-by-step guide to running the full toolkit end to end, from a fresh clone to working notebooks.

---

## Prerequisites

- Python 3.11+
- Node not required (everything is Python / SQL)
- A BigQuery project is **optional** — the entire toolkit runs on synthetic data without a warehouse

---

## Step 1 — Set up the environment

```bash
git clone https://github.com/violaciao/data-bedrock.git
cd data-bedrock

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Verify:
```bash
python -c "import pandas, scipy, sklearn, matplotlib; print('OK')"
```

---

## Step 2 — Configure environment variables

```bash
cp .env.example .env
```

For synthetic-only use, no edits are needed. If you later connect a real warehouse, fill in your BigQuery credentials here. Never commit `.env`.

---

## Step 3 — Generate synthetic data

Everything downstream depends on this.

```bash
python data/synthetic/generate_synthetic_data.py
```

This produces three files under `data/synthetic/`:

| File | Rows | What it represents |
|------|------|-------------------|
| `users.csv` | 5,000 | Signup cohorts across 12 months, 6 acquisition channels, ~5% monthly churn |
| `events.csv` | ~52,000 | Product events with Zipf-distributed frequency per user |
| `orders.csv` | ~2,600 | Revenue events tied to users |

These files are gitignored — regenerate them any time with the same script.

---

## Step 4 — Run the test suite

Confirms all statistical and analytics modules are working before you open any notebook.

```bash
pytest tests/ -v
```

Expected: **112 passed**. The suite covers:
- A/B testing stats, sample size, sequential testing
- Cohort retention matrix
- Funnel analysis
- Difference-in-Differences, PSM, bandits, MMM

---

## Step 5 — Explore growth analysis

**Funnel conversion**

```bash
jupyter notebook 03_growth_analysis/funnel_analysis.ipynb
```

What you will see:
- Top-of-funnel users → conversion step by step
- Drop-off rates at each stage
- Conversion from previous step vs. from top

**Acquisition channels**

```bash
jupyter notebook 03_growth_analysis/acquisition_analysis.ipynb
```

What you will see:
- Users by acquisition channel
- Conversion rate differences across channels
- Revenue and LTV by channel

---

## Step 6 — Explore cohort retention

**Cohort heatmap**

```bash
jupyter notebook 04_cohort_retention/cohort_heatmap.ipynb
```

What you will see:
- Retention matrix (cohort week × period number)
- Colour-coded heatmap — darker = better retention
- Period 0 is always 100%; watch for the cliff at period 1–2

**Retention curves**

```bash
jupyter notebook 04_cohort_retention/retention_curves.ipynb
```

What you will see:
- Average retention curve across all cohorts
- Power-law curve fit with R² and long-run retention estimate
- Retention curves broken out by acquisition channel

---

## Step 7 — Run experiments and causal inference

```bash
jupyter notebook 05_measurement/experiment_analysis.ipynb
```

What you will see:
- z-test and Welch t-test on synthetic A/B groups
- CUPED variance reduction
- Sample size calculator
- Sequential test (mSPRT) — always-valid p-values

For causal inference, use the library directly in a notebook or script:

```python
import sys; sys.path.insert(0, "05_measurement")

# Difference-in-Differences
from causal_inference.did import DifferenceInDifferences
did = DifferenceInDifferences()
result = did.estimate_2x2(pre_treated, post_treated, pre_control, post_control)
print(result)

# Propensity Score Matching
from causal_inference.psm import PropensityScoreMatching
psm = PropensityScoreMatching(caliper=0.05)
result = psm.match(df, "treated", "revenue", ["age", "tenure"])
print(result.balance)   # check SMD before/after

# Thompson Sampling bandit
from causal_inference.bandit import ThompsonSampling
bandit = ThompsonSampling(n_arms=3)
arm = bandit.select_arm()
bandit.update(arm, reward=1.0)

# Marketing Mix Model
from causal_inference.mmm import MarketingMixModel, ChannelConfig
mmm = MarketingMixModel(channel_configs=[
    ChannelConfig("paid_search", decay=0.3, alpha=2.0, K=1000),
    ChannelConfig("social",      decay=0.5, alpha=1.5, K=500),
])
result = mmm.fit(df, "revenue")
print(result)
mmm.optimize_budget(total_budget=50_000)
```

---

## Step 8 — Run dbt models (optional, requires BigQuery)

```bash
cd 01_dbt_project
cp profiles.yml.example profiles.yml
# Edit profiles.yml with your GCP project and dataset
dbt debug          # confirms connection
dbt run            # builds all models
dbt test           # runs schema + data tests
dbt docs generate && dbt docs serve   # browse lineage in browser
```

Key models:

| Model | What it produces |
|-------|-----------------|
| `mart_cohort_retention` | Cohort × period retention matrix |
| `mart_mrr` | New / expansion / contraction / churned MRR by week |
| `fct_funnel` | Step-by-step funnel conversion |
| `dim_users` | User dimension with acquisition channel and plan |

Without a warehouse, the Python modules (`cohort_retention/`, `growth_analysis/`) replicate this logic directly on the CSV files.

---

## Step 9 — Review the tracking plan and metrics dictionary

Before adapting the repo to a real product:

1. **`00_tracking_plan/tracking_plan.md`** — review the event taxonomy and decide which events map to your product
2. **`00_tracking_plan/segment_schema.json`** — Segment-compatible JSON schema, ready to import
3. **`02_metrics_dictionary/`** — canonical definitions for DAU, MAU, MRR, retention, LTV; adapt names to match your team's language before anyone builds a dashboard

---

## Step 10 — Dashboards

```bash
cat 06_dashboards/README.md
```

- **Metabase**: import the JSON exports from `06_dashboards/metabase/` into a running Metabase instance; point them at your mart tables
- **Looker Studio**: follow `06_dashboards/looker_studio_guide.md` to connect BigQuery and replicate the layouts

---

## Recommended order for a real project

```
Day 1   Steps 1–4         Environment, synthetic data, tests green
Day 2   Steps 5–6         Explore growth and retention patterns on synthetic data
Day 3   Step 7            Run experiment analysis; pick the causal methods relevant to your use cases
Day 4   Step 9            Adapt tracking plan and metrics dictionary to your product
Day 5   Step 8            Connect dbt to your warehouse; run models on real data
Day 6   Step 10           Stand up dashboards against mart tables
```

---

## Quick reference

```bash
# Re-run everything from scratch
python data/synthetic/generate_synthetic_data.py
pytest tests/ -v
jupyter notebook

# Lint
ruff check .

# Single test file
pytest tests/test_stats.py -v
pytest tests/test_mmm.py -v
```
