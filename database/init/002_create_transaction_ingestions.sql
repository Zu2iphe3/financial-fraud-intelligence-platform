BEGIN;

CREATE TABLE IF NOT EXISTS raw.transaction_ingestions (
    event_id UUID PRIMARY KEY,
    transaction_id UUID NOT NULL REFERENCES raw.transactions (transaction_id),
    event_version VARCHAR(8) NOT NULL,
    kafka_topic TEXT NOT NULL,
    kafka_partition INTEGER NOT NULL CHECK (kafka_partition >= 0),
    kafka_offset BIGINT NOT NULL CHECK (kafka_offset >= 0),
    published_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT transaction_ingestions_topic_position_key
        UNIQUE (kafka_topic, kafka_partition, kafka_offset)
);

COMMENT ON TABLE raw.transaction_ingestions IS
    'Kafka delivery metadata kept separately from the logical PaySim transaction.';
COMMENT ON COLUMN raw.transaction_ingestions.event_id IS
    'Stable identity for one versioned PaySim replay event; repeated delivery is idempotent.';
COMMENT ON COLUMN raw.transaction_ingestions.transaction_id IS
    'Logical PaySim row identity shared with batch ingestion.';

CREATE INDEX IF NOT EXISTS transaction_ingestions_transaction_id_idx
    ON raw.transaction_ingestions (transaction_id);

COMMIT;
