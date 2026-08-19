{{ config(tags=['transactions', 'mart']) }}

select
    transaction_id,
    source_dataset_id,
    source_row_number,
    step,
    simulation_day,
    simulation_hour_of_day,
    transaction_type,
    amount,
    name_orig,
    name_dest,
    old_balance_orig,
    new_balance_orig,
    old_balance_dest,
    new_balance_dest,
    is_fraud,
    is_fraudulent,
    is_flagged_fraud,
    is_flagged_fraudulent,
    first_ingestion_source,
    source_file_name,
    ingested_at
from {{ ref('int_transactions_enriched') }}
