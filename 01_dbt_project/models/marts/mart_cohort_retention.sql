-- mart_cohort_retention: cohort_week × period_number → retention_rate
--
-- Output schema:
--   cohort_week        DATE        The week users signed up
--   period_number      INT64       Weeks since signup (0 = signup week)
--   cohort_size        INT64       Number of users in the cohort
--   active_users       INT64       Users still active at this period
--   retention_rate     FLOAT64     active_users / cohort_size

with

spine as (
    select
        user_id,
        cohort_week,
        week_start,
        periods_since_signup

    from {{ ref('int_user_spine') }}
),

activity as (
    select
        user_id,
        week_start

    from {{ ref('int_event_counts') }}
    where is_active = true
),

-- Join spine to activity: was user active in this week?
spine_with_activity as (
    select
        s.user_id,
        s.cohort_week,
        s.periods_since_signup,
        s.week_start,
        coalesce(a.user_id is not null, false)                  as was_active

    from spine s
    left join activity a
        on s.user_id   = a.user_id
        and s.week_start = a.week_start
),

-- Cohort size = distinct users in period 0
cohort_sizes as (
    select
        cohort_week,
        count(distinct user_id)                                 as cohort_size

    from spine_with_activity
    where periods_since_signup = 0
    group by cohort_week
),

-- Active users per cohort × period
cohort_activity as (
    select
        cohort_week,
        periods_since_signup                                    as period_number,
        count(distinct case when was_active then user_id end)   as active_users

    from spine_with_activity
    group by cohort_week, periods_since_signup
),

final as (
    select
        ca.cohort_week,
        ca.period_number,
        cs.cohort_size,
        ca.active_users,
        safe_divide(ca.active_users, cs.cohort_size)            as retention_rate

    from cohort_activity ca
    inner join cohort_sizes cs
        on ca.cohort_week = cs.cohort_week
)

select * from final
order by cohort_week, period_number
