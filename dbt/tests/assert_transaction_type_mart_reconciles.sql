with transaction_metrics as (
    select
        count(*)::bigint as transaction_count,
        sum(amount)::numeric as transaction_amount,
        sum(is_fraud)::bigint as fraud_count,
        sum(is_flagged_fraud)::bigint as flagged_fraud_count
    from {{ ref('fct_transactions') }}
),
type_metrics as (
    select
        sum(transaction_count)::bigint as transaction_count,
        sum(transaction_amount)::numeric as transaction_amount,
        sum(fraud_count)::bigint as fraud_count,
        sum(flagged_fraud_count)::bigint as flagged_fraud_count
    from {{ ref('mart_fraud_by_transaction_type') }}
)

select type_metrics.*
from type_metrics
cross join transaction_metrics
where type_metrics.transaction_count is distinct from transaction_metrics.transaction_count
   or type_metrics.transaction_amount is distinct from transaction_metrics.transaction_amount
   or type_metrics.fraud_count is distinct from transaction_metrics.fraud_count
   or type_metrics.flagged_fraud_count is distinct from transaction_metrics.flagged_fraud_count
