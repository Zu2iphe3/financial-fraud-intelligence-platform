{{ config(tags=['fraud_monitoring', 'mart']) }}

with type_measures as (
    select
        transaction_type,
        count(*)::bigint as transaction_count,
        sum(amount)::numeric(30, 2) as transaction_amount,
        count(*) filter (where is_fraudulent)::bigint as fraud_count,
        coalesce(sum(amount) filter (where is_fraudulent), 0)::numeric(30, 2)
            as fraud_amount,
        count(*) filter (where is_flagged_fraudulent)::bigint as flagged_fraud_count
    from {{ ref('fct_transactions') }}
    group by transaction_type
)

select
    transaction_type,
    transaction_count,
    transaction_amount,
    fraud_count,
    fraud_amount,
    flagged_fraud_count,
    (fraud_count::numeric / nullif(transaction_count, 0))::numeric(18, 10)
        as fraud_rate
from type_measures
