-- mart_mrr: monthly MRR movements.
--
-- Classifies each billing month into MRR categories:
--   new_mrr        : first payment from this user
--   expansion_mrr  : upgrade to a higher plan
--   contraction_mrr: downgrade to a lower plan
--   churned_mrr    : last payment before cancellation (negative contribution)
--   renewal_mrr    : recurring payment at the same plan
--
-- One row per (user_id, billing_month). The union approach lets each category
-- be debugged independently before rolling up.

with

orders as (
    select
        order_id,
        user_id,
        order_type,
        plan,
        amount_usd,
        date_trunc(billed_at, month)                            as billing_month,
        billed_at

    from {{ ref('stg_orders') }}
    where order_type != 'churn'   -- churn rows carry $0; we derive churned_mrr below
),

-- Assign a plan rank so we can detect upgrades vs downgrades
plan_ranks as (
    select 'free'       as plan, 0 as plan_rank union all
    select 'starter',   1 union all
    select 'growth',    2 union all
    select 'enterprise',3
),

orders_with_rank as (
    select
        o.*,
        pr.plan_rank

    from orders o
    left join plan_ranks pr on o.plan = pr.plan
),

-- Prior month's amount per user
with_prior as (
    select
        *,
        lag(amount_usd)  over (partition by user_id order by billing_month) as prior_amount,
        lag(plan_rank)   over (partition by user_id order by billing_month) as prior_plan_rank,
        row_number()     over (partition by user_id order by billing_month) as month_number

    from orders_with_rank
),

-- Classify MRR type
classified as (
    select
        order_id,
        user_id,
        billing_month,
        plan,
        amount_usd,

        case
            when month_number = 1                               then 'new_mrr'
            when plan_rank > coalesce(prior_plan_rank, 0)       then 'expansion_mrr'
            when plan_rank < prior_plan_rank                    then 'contraction_mrr'
            else                                                     'renewal_mrr'
        end                                                     as mrr_type,

        -- Net MRR delta vs prior month
        amount_usd - coalesce(prior_amount, 0)                  as mrr_delta

    from with_prior
),

-- Churned MRR: users whose last order was followed by a churn row
churned_users as (
    select
        user_id,
        date_trunc(billed_at, month)                            as churn_month,
        -- Use negative of their last known MRR
        -1 * amount_usd                                         as churned_amount

    from {{ ref('stg_orders') }}
    where order_type = 'churn'
),

churned_mrr as (
    select
        null                                                    as order_id,
        c.user_id,
        c.churn_month                                           as billing_month,
        o.plan,
        c.churned_amount                                        as amount_usd,
        'churned_mrr'                                           as mrr_type,
        c.churned_amount                                        as mrr_delta

    from churned_users c
    left join (
        select user_id, plan, billed_at,
               row_number() over (partition by user_id order by billed_at desc) as rn
        from {{ ref('stg_orders') }}
        where order_type != 'churn'
    ) o on c.user_id = o.user_id and o.rn = 1
),

final as (
    select * from classified
    union all
    select * from churned_mrr
)

select
    billing_month,
    user_id,
    order_id,
    plan,
    mrr_type,
    amount_usd,
    mrr_delta

from final
order by billing_month, user_id
