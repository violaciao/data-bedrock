-- dim_users: user dimension table with enriched attributes.
--
-- Denormalises useful aggregates (lifetime events, first/last activity) onto
-- the user record so mart queries don't need to re-join events repeatedly.

with

users as (
    select * from {{ ref('stg_users') }}
),

event_summary as (
    select
        user_id,
        min(occurred_at)                                        as first_event_at,
        max(occurred_at)                                        as last_event_at,
        count(*)                                                as lifetime_events,
        count(distinct date_trunc(occurred_at, week))           as active_weeks

    from {{ ref('stg_events') }}
    group by user_id
),

final as (
    select
        u.user_id,
        u.signed_up_at,
        u.acquisition_channel,
        u.is_converted,
        u.plan,
        u.churned_at,
        u.country_code,

        -- Activity metrics
        es.first_event_at,
        es.last_event_at,
        es.lifetime_events,
        es.active_weeks,

        -- Derived flags
        u.churned_at is not null                                as is_churned,
        u.plan in ('starter', 'growth', 'enterprise')           as is_paying,

        -- Time-to-convert (days from signup to first paid order, if converted)
        date_diff(
            cast(
                (select min(billed_at)
                 from {{ ref('stg_orders') }} o
                 where o.user_id = u.user_id
                   and o.order_type = 'new_mrr'
                ) as date
            ),
            cast(u.signed_up_at as date),
            day
        )                                                       as days_to_convert

    from users u
    left join event_summary es on u.user_id = es.user_id
)

select * from final
