{{ config(tags=['fraud_monitoring', 'mart']) }}

with daily_measures as (
    select
        simulation_day,
        min(step)::integer as first_step,
        max(step)::integer as last_step,
        count(distinct step)::integer as hours_observed,
        count(*)::bigint as transaction_count,
        sum(amount)::numeric(30, 2) as transaction_amount,
        count(*) filter (where is_fraudulent)::bigint as fraud_count,
        coalesce(sum(amount) filter (where is_fraudulent), 0)::numeric(30, 2)
            as fraud_amount,
        count(*) filter (where is_flagged_fraudulent)::bigint as flagged_fraud_count
    from {{ ref('fct_transactions') }}
    group by simulation_day
)

select
    simulation_day,
    first_step,
    last_step,
    hours_observed,
    (hours_observed < 24) as is_partial_day,
    transaction_count,
    transaction_amount,
    fraud_count,
    fraud_amount,
    flagged_fraud_count,
    (fraud_count::numeric / nullif(transaction_count, 0))::numeric(18, 10)
        as fraud_rate
from daily_measures
