with raw_metrics as (
    select
        count(*)::bigint as transaction_count,
        sum(amount)::numeric as transaction_amount,
        sum(is_fraud)::bigint as fraud_count,
        sum(is_flagged_fraud)::bigint as flagged_fraud_count
    from {{ source('raw', 'transactions') }}
),
staging_metrics as (
    select
        count(*)::bigint as transaction_count,
        sum(amount)::numeric as transaction_amount,
        sum(is_fraud)::bigint as fraud_count,
        sum(is_flagged_fraud)::bigint as flagged_fraud_count
    from {{ ref('stg_transactions') }}
)

-- A dbt singular test passes only when this comparison returns zero rows.
select staging_metrics.*
from staging_metrics
cross join raw_metrics
where staging_metrics.transaction_count is distinct from raw_metrics.transaction_count
   or staging_metrics.transaction_amount is distinct from raw_metrics.transaction_amount
   or staging_metrics.fraud_count is distinct from raw_metrics.fraud_count
   or staging_metrics.flagged_fraud_count is distinct from raw_metrics.flagged_fraud_count
