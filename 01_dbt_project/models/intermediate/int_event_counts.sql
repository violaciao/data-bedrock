-- int_event_counts: aggregate event counts per user per week.
--
-- Materialised as ephemeral so it compiles inline into mart queries.
-- Used by fct_funnel and mart_cohort_retention to determine "active" status.

with

events as (
    select
        user_id,
        event_type,
        date_trunc(occurred_at, week)                           as week_start,
        occurred_at

    from {{ ref('stg_events') }}
),

weekly_counts as (
    select
        user_id,
        week_start,
        countif(event_type = 'page_view')                       as page_views,
        countif(event_type = 'login')                           as logins,
        countif(event_type = 'feature_used')                    as feature_uses,
        countif(event_type = 'report_created')                  as reports_created,
        countif(event_type = 'export_downloaded')               as exports,
        count(*)                                                as total_events

    from events
    group by user_id, week_start
),

-- A user is "active" in a week if they had at least 1 non-page-view event
-- (page views alone may be bots or accidental visits)
with_activity_flag as (
    select
        *,
        (total_events - page_views) > 0                         as is_active

    from weekly_counts
)

select * from with_activity_flag
