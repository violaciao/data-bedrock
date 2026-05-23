-- int_user_spine: one row per user per calendar week, from signup to churn (or today).
--
-- This is the backbone for cohort and retention analysis. Every mart model that
-- needs "was user X active in week W?" should join against this spine rather than
-- re-deriving it from events — that keeps the logic DRY and the definitions
-- consistent.
--
-- The date spine is built here (not in marts) per Architecture Rule 4.

with

users as (
    select
        user_id,
        date_trunc(signed_up_at, week)                          as cohort_week,
        signed_up_at,
        coalesce(churned_at, current_timestamp())               as active_through,
        is_converted,
        plan,
        acquisition_channel

    from {{ ref('stg_users') }}
),

-- Generate one row per week between signup and churn/today
date_spine as (
    {{ dbt_utils.date_spine(
        datepart       = "week",
        start_date     = "cast('2024-01-01' as date)",
        end_date       = "current_date()"
    ) }}
),

spine as (
    select
        u.user_id,
        u.cohort_week,
        u.signed_up_at,
        u.is_converted,
        u.plan,
        u.acquisition_channel,
        cast(d.date_week as timestamp)                          as week_start,
        date_diff(cast(d.date_week as date), cast(u.cohort_week as date), week)
                                                                as periods_since_signup

    from users u
    cross join date_spine d
    where d.date_week >= cast(u.cohort_week as date)
      and d.date_week <  cast(date_trunc(u.active_through, week) as date)
)

select * from spine
