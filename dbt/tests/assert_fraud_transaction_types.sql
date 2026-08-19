select transaction_type, fraud_count
from {{ ref('mart_fraud_by_transaction_type') }}
where fraud_count > 0
  and transaction_type not in ('CASH_OUT', 'TRANSFER')
