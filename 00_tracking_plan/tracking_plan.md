# Tracking Plan

Event taxonomy for the data-bedrock SaaS application. All events must be
instrumented via Segment (or equivalent CDP) before being used in analysis.

**Owner:** Data team
**Last reviewed:** May 2026

---

## Instrumentation rules

1. Every event must have `user_id` (after signup) or `anonymous_id` (pre-signup).
2. All timestamps must be UTC.
3. String properties must be lower-cased and snake_cased at the SDK level.
4. Never log PII in event properties (names, emails, raw text input).

---

## Events

### `page_view`
Fired on every page render.

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `page_name` | string | ✓ | Slug of the page (e.g. `dashboard`, `settings`) |
| `referrer` | string | | Previous page URL |
| `utm_source` | string | | UTM source if present |
| `utm_medium` | string | | UTM medium if present |
| `utm_campaign` | string | | UTM campaign if present |

---

### `signup`
Fired when a user completes account creation.

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `acquisition_channel` | string | ✓ | Attribution channel (organic_search / paid_search / direct / referral / social / email) |
| `signup_method` | string | ✓ | email / google / github |
| `plan_at_signup` | string | ✓ | Plan selected at signup (free / starter / growth / enterprise) |

---

### `login`
Fired on successful authentication.

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `login_method` | string | ✓ | email / google / github / sso |

---

### `feature_used`
Fired when a user interacts with a core product feature.

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `feature_name` | string | ✓ | Snake-cased feature identifier |
| `feature_category` | string | | Grouping category |

---

### `dashboard_viewed`
Fired when a user opens a dashboard.

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `dashboard_id` | string | ✓ | |
| `dashboard_type` | string | | built_in / custom |

---

### `report_created`
Fired when a user saves a new report.

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `report_type` | string | ✓ | funnel / cohort / ab_test / custom |
| `data_sources` | array | | List of data sources used |

---

### `invite_sent`
Fired when a user invites a teammate.

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `invite_role` | string | ✓ | viewer / editor / admin |

---

### `upgrade_clicked`
Fired when a user clicks any upgrade CTA.

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `cta_location` | string | ✓ | Where in the UI the CTA appeared |
| `current_plan` | string | ✓ | User's plan at time of click |
| `target_plan` | string | | Plan shown in the upgrade modal |

---

### `export_downloaded`
Fired when a user downloads data or a report.

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `export_format` | string | ✓ | csv / xlsx / png / pdf |
| `export_type` | string | ✓ | report / raw_data / chart |

---

### `settings_changed`
Fired when a user saves a settings change.

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `setting_name` | string | ✓ | Name of the changed setting |

---

## Identify calls

The `identify` call must be fired:
1. Immediately after signup (with `plan`, `acquisition_channel`)
2. After any plan change (update `plan` trait)

**Required traits:** `plan`, `acquisition_channel`, `signed_up_at`, `country`
**Forbidden traits:** email, name, raw PII
