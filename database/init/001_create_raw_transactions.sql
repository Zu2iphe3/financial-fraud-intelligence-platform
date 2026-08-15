BEGIN;

CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.transactions (
    record_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- The logical identity is stable across batch loading and later Kafka replay.
    transaction_id UUID NOT NULL,
    source_dataset_id CHAR(64) NOT NULL,
    source_row_number BIGINT NOT NULL CHECK (source_row_number > 0),

    step INTEGER NOT NULL CHECK (step > 0),
    transaction_type VARCHAR(16) NOT NULL CHECK (
        transaction_type IN ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER')
    ),
    amount NUMERIC(20, 2) NOT NULL CHECK (amount >= 0),
    name_orig TEXT NOT NULL,
    old_balance_orig NUMERIC(20, 2) NOT NULL CHECK (old_balance_orig >= 0),
    new_balance_orig NUMERIC(20, 2) NOT NULL CHECK (new_balance_orig >= 0),
    name_dest TEXT NOT NULL,
    old_balance_dest NUMERIC(20, 2) NOT NULL CHECK (old_balance_dest >= 0),
    new_balance_dest NUMERIC(20, 2) NOT NULL CHECK (new_balance_dest >= 0),
    is_fraud SMALLINT NOT NULL CHECK (is_fraud IN (0, 1)),
    is_flagged_fraud SMALLINT NOT NULL CHECK (is_flagged_fraud IN (0, 1)),

    first_ingestion_source VARCHAR(16) NOT NULL CHECK (
        first_ingestion_source IN ('batch', 'kafka')
    ),
    source_file_name TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT transactions_transaction_id_key UNIQUE (transaction_id),
    CONSTRAINT transactions_dataset_row_key UNIQUE (source_dataset_id, source_row_number),
    CONSTRAINT transactions_source_dataset_id_format_check CHECK (
        source_dataset_id ~ '^[0-9a-f]{64}$'
    )
);

COMMENT ON TABLE raw.transactions IS
    'PaySim source transactions plus separately named platform ingestion metadata.';
COMMENT ON COLUMN raw.transactions.transaction_id IS
    'Path-independent UUIDv5 derived from the exact dataset SHA-256 and original one-based data row.';
COMMENT ON COLUMN raw.transactions.source_dataset_id IS
    'Lowercase SHA-256 fingerprint of the exact PaySim CSV bytes.';
COMMENT ON COLUMN raw.transactions.source_row_number IS
    'Original one-based PaySim data-row number, excluding the CSV header.';
COMMENT ON COLUMN raw.transactions.old_balance_orig IS
    'Analysis/audit only: simulator leakage field prohibited as a predictive feature.';
COMMENT ON COLUMN raw.transactions.new_balance_orig IS
    'Analysis/audit only: simulator leakage field prohibited as a predictive feature.';
COMMENT ON COLUMN raw.transactions.old_balance_dest IS
    'Analysis/audit only: simulator leakage field prohibited as a predictive feature.';
COMMENT ON COLUMN raw.transactions.new_balance_dest IS
    'Analysis/audit only: simulator leakage field prohibited as a predictive feature.';

CREATE INDEX IF NOT EXISTS transactions_step_idx
    ON raw.transactions (step);
CREATE INDEX IF NOT EXISTS transactions_type_step_idx
    ON raw.transactions (transaction_type, step);
CREATE INDEX IF NOT EXISTS transactions_fraud_idx
    ON raw.transactions (step, transaction_type)
    WHERE is_fraud = 1;

COMMIT;
