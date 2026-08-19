select transaction_id, simulation_day, simulation_hour_of_day
from {{ ref('int_transactions_enriched') }}
where simulation_day < 1
   or simulation_hour_of_day not between 1 and 24
