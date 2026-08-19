with transaction_metrics as (
    select
        count(*)::bigint as transaction_count,
        sum(amount)::numeric as transaction_amount,
        sum(is_fraud)::bigint as fraud_count,
        sum(is_flagged_fraud)::bigint as flagged_fraud_count
    from {{ ref('fct_transactions') }}
),
daily_metrics as (
    select
        sum(transaction_count)::bigint as transaction_count,
        sum(transaction_amount)::numeric as transaction_amount,
        sum(fraud_count)::bigint as fraud_count,
        sum(flagged_fraud_count)::bigint as flagged_fraud_count
    from {{ ref('mart_fraud_by_day') }}
)

select daily_metrics.*
from daily_metrics
cross join transaction_metrics
where daily_metrics.transaction_count is distinct from transaction_metrics.transaction_count
   or daily_metrics.transaction_amount is distinct from transaction_metrics.transaction_amount
   or daily_metrics.fraud_count is distinct from transaction_metrics.fraud_count
   or daily_metrics.flagged_fraud_count is distinct from transaction_metrics.flagged_fraud_count
