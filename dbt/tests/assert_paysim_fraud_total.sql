select sum(fraud_count)::bigint as fraud_count
from {{ ref('mart_fraud_by_transaction_type') }}
having sum(fraud_count)::bigint != 8213
