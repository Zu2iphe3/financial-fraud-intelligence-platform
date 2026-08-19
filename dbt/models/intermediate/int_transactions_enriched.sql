{{ config(tags=['transactions', 'intermediate']) }}

select
    *,
    -- PaySim step 1 is the first simulated hour, so subtract before grouping by 24.
    (((step - 1) / 24) + 1)::integer as simulation_day,
    (mod(step - 1, 24) + 1)::integer as simulation_hour_of_day,
    (is_fraud = 1) as is_fraudulent,
    (is_flagged_fraud = 1) as is_flagged_fraudulent
from {{ ref('stg_transactions') }}
