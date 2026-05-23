-- stg_events: clean and deduplicate the raw event stream.
--
-- Notes:
--   - We keep duplicate event_ids in case of ingestion retries — dedup happens
--     in intermediate models once we know the dedup key.
--   - properties is kept as a raw JSON string; parse it in intermediate models
--     for specific event types.

with

source as (
    select * from {{ source('raw', 'events') }}
),

renamed as (
    select
        event_id,
        user_id,
        lower(trim(event_type))                                 as event_type,
        cast(occurred_at as timestamp)                          as occurred_at,
        properties

    from source
    where event_id is not null
      and user_id  is not null
      and occurred_at is not null
)

select * from renamed
