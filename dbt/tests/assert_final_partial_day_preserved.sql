with expected_final_day as (
    select 31::integer as simulation_day
)

select expected_final_day.simulation_day
from expected_final_day
left join {{ ref('mart_fraud_by_day') }} as actual
    on expected_final_day.simulation_day = actual.simulation_day
   and actual.hours_observed = 23
   and actual.is_partial_day
   and actual.transaction_count = 272
   and actual.fraud_count = 272
   and actual.fraud_rate = 1
where actual.simulation_day is null
