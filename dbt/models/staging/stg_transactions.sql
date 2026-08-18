{{ config(tags=['transactions', 'staging']) }}

select
    record_id::bigint as record_id,
    transaction_id::uuid as transaction_id,
    source_dataset_id::text as source_dataset_id,
    source_row_number::bigint as source_row_number,
    step::integer as step,
    transaction_type::varchar(16) as transaction_type,
    amount::numeric(20, 2) as amount,
    name_orig::text as name_orig,
    old_balance_orig::numeric(20, 2) as old_balance_orig,
    new_balance_orig::numeric(20, 2) as new_balance_orig,
    name_dest::text as name_dest,
    old_balance_dest::numeric(20, 2) as old_balance_dest,
    new_balance_dest::numeric(20, 2) as new_balance_dest,
    is_fraud::smallint as is_fraud,
    is_flagged_fraud::smallint as is_flagged_fraud,
    first_ingestion_source::varchar(16) as first_ingestion_source,
    source_file_name::text as source_file_name,
    ingested_at::timestamptz as ingested_at
from {{ source('raw', 'transactions') }}
