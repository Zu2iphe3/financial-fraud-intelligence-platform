BEGIN;

-- Point-in-time inference filters by one dataset and destination before applying
-- strict PaySim step ordering. Without this index, every score can scan 6.3M rows.
CREATE INDEX IF NOT EXISTS transactions_dataset_destination_step_row_idx
    ON raw.transactions (source_dataset_id, name_dest, step, source_row_number);

COMMIT;
