-- fct_funnel: parameterized funnel conversion.
--
-- The funnel steps are defined in seeds/funnel_config.csv rather than
-- hardcoded here. This allows stakeholders to change the funnel definition
-- without touching SQL.
--
-- Output: one row per (funnel_name, step_order) with user counts and
-- step-over-step conversion rates.
--
-- To add a new funnel:
--   1. Add rows to seeds/funnel_config.csv
--   2. Run: dbt seed && dbt run --select fct_funnel

with

config as (
    select * from {{ ref('funnel_config') }}
),

events as (
    select
        user_id,
        event_type,
        min(occurred_at)                                        as first_occurred_at

    from {{ ref('stg_events') }}
    group by user_id, event_type
),

-- For each funnel step, find users who completed it
step_users as (
    select
        c.funnel_name,
        c.step_order,
        c.step_name,
        c.event_type,
        e.user_id,
        e.first_occurred_at

    from config c
    inner join events e on c.event_type = e.event_type
),

-- For each user × funnel, find the maximum step they completed in order
-- (user must complete step N before step N+1 counts)
ordered_completions as (
    select
        su.funnel_name,
        su.user_id,
        su.step_order,
        su.step_name,
        su.event_type,

        -- Did this user complete ALL prior steps?
        (
            select count(distinct prior.step_order)
            from step_users prior
            where prior.funnel_name = su.funnel_name
              and prior.user_id     = su.user_id
              and prior.step_order  < su.step_order
        ) = su.step_order - (
            select min(min_step.step_order)
            from config min_step
            where min_step.funnel_name = su.funnel_name
        )                                                       as completed_in_order

    from step_users su
),

funnel_counts as (
    select
        funnel_name,
        step_order,
        step_name,
        event_type,
        countif(completed_in_order)                             as users_reached

    from ordered_completions
    group by funnel_name, step_order, step_name, event_type
),

with_conversion as (
    select
        funnel_name,
        step_order,
        step_name,
        event_type,
        users_reached,

        -- Conversion from top of funnel
        safe_divide(
            users_reached,
            first_value(users_reached) over (
                partition by funnel_name order by step_order
            )
        )                                                       as conversion_from_top,

        -- Step-over-step conversion
        safe_divide(
            users_reached,
            lag(users_reached) over (partition by funnel_name order by step_order)
        )                                                       as conversion_from_previous

    from funnel_counts
)

select * from with_conversion
order by funnel_name, step_order
