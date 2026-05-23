# Synthetic Data

Example SaaS event data for local development. Generated files are gitignored.

## Generate

```bash
python data/synthetic/generate_synthetic_data.py
```

## Schema

### users.csv

| Column | Type | Description |
|--------|------|-------------|
| user_id | string | `user_00001` … |
| signup_at | timestamp | Signup timestamp |
| acquisition_channel | string | organic_search / paid_search / direct / referral / social / email |
| converted | bool | Whether the user ever paid |
| plan | string | free / starter / growth / enterprise |
| churn_at | timestamp | Churn date (null if still active) |
| country | string | ISO 2-letter country code |

### events.csv

| Column | Type | Description |
|--------|------|-------------|
| event_id | string | `evt_0000001` … |
| user_id | string | FK to users |
| event_type | string | page_view / signup / login / feature_used / … |
| occurred_at | timestamp | Event timestamp |
| properties | json_string | `{}` placeholder |

### orders.csv

| Column | Type | Description |
|--------|------|-------------|
| order_id | string | `ord_000001` … |
| user_id | string | FK to users |
| order_type | string | new_mrr / renewal / churn |
| plan | string | starter / growth / enterprise |
| amount_usd | int | Monthly charge in USD |
| billed_at | timestamp | Billing date |
| period_start | timestamp | Subscription period start |
| period_end | timestamp | Subscription period end |

## Distributions

- **5 000 users** signed up between 2024-01-01 and 2024-12-31 with exponential growth bias
- **Acquisition channels** have different conversion rates (email 40% → social 12%)
- **Event frequency** follows a Zipf(1.8) distribution (power-law)
- **Churn rate** ~5% monthly, producing a realistic survival curve
- **`random.seed(42)`** — outputs are fully deterministic
