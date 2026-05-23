-- stg_orders: clean the raw orders / subscription payments table.

with

source as (
    select * from {{ source('raw', 'orders') }}
),

renamed as (
    select
        order_id,
        user_id,
        lower(trim(order_type))                                 as order_type,
        lower(trim(plan))                                       as plan,
        cast(amount_usd as numeric)                             as amount_usd,
        cast(billed_at as timestamp)                            as billed_at,
        cast(period_start as timestamp)                         as period_start,
        cast(period_end as timestamp)                           as period_end

    from source
    where order_id is not null
)

select * from renamed
