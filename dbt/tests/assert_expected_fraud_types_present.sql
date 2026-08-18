with expected_types as (
    select unnest(array['CASH_OUT', 'TRANSFER']) as transaction_type
)

select expected_types.transaction_type
from expected_types
left join {{ ref('mart_fraud_by_transaction_type') }} as actual
    on expected_types.transaction_type = actual.transaction_type
   and actual.fraud_count > 0
where actual.transaction_type is null
