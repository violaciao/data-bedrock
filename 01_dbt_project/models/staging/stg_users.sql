-- stg_users: light cleaning of the raw users table.
-- Source: raw.users (loaded from data/synthetic/users.csv via your ingestion tool)
--
-- Transformations:
--   - Rename raw columns to snake_case standard
--   - Cast timestamps
--   - No business logic here — that belongs in intermediate or mart models

with

source as (
    select * from {{ source('raw', 'users') }}
),

renamed as (
    select
        user_id,
        cast(signup_at as timestamp)                            as signed_up_at,
        acquisition_channel,
        cast(converted as bool)                                 as is_converted,
        lower(trim(plan))                                       as plan,
        cast(churn_at as timestamp)                             as churned_at,
        upper(trim(country))                                    as country_code

    from source
)

select * from renamed
