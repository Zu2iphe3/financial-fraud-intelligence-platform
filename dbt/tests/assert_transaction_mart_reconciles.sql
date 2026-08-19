with intermediate_metrics as (
    select
        count(*)::bigint as transaction_count,
        sum(amount)::numeric as transaction_amount,
        sum(is_fraud)::bigint as fraud_count,
        sum(is_flagged_fraud)::bigint as flagged_fraud_count
    from {{ ref('int_transactions_enriched') }}
),
mart_metrics as (
    select
        count(*)::bigint as transaction_count,
        sum(amount)::numeric as transaction_amount,
        sum(is_fraud)::bigint as fraud_count,
        sum(is_flagged_fraud)::bigint as flagged_fraud_count
    from {{ ref('fct_transactions') }}
)

select mart_metrics.*
from mart_metrics
cross join intermediate_metrics
where mart_metrics.transaction_count is distinct from intermediate_metrics.transaction_count
   or mart_metrics.transaction_amount is distinct from intermediate_metrics.transaction_amount
   or mart_metrics.fraud_count is distinct from intermediate_metrics.fraud_count
   or mart_metrics.flagged_fraud_count is distinct from intermediate_metrics.flagged_fraud_count
