# Kafka transaction replay pipeline

## Purpose

Phase 4 adds a local real-time path alongside the Phase 3 batch loader:

```text
PaySim CSV -> Python producer -> Apache Kafka -> Python consumer -> PostgreSQL
```

The producer replays a chosen PaySim row range at a configurable speed. Kafka holds
those events until the consumer is ready. The consumer validates each event and stores
it in the same `raw.transactions` table used by batch ingestion. No parallel streaming
transaction store is introduced.

## Identity and duplicate policy

Two identities serve different purposes:

- `transaction_id` identifies the original PaySim row. It is derived from the exact CSV
  SHA-256 fingerprint and original one-based row number, so batch ingestion and Kafka
  replay produce the same value.
- `event_id` identifies the versioned replay event for that transaction. It is stable
  when the same row is replayed again.

PostgreSQL enforces one logical transaction per `transaction_id`. The separate
`raw.transaction_ingestions` table retains the first Kafka topic, partition, offset,
event version, and timestamps for each `event_id`. Replaying or redelivering an event
therefore does not create a second transaction or a second event audit record.

If Phase 3 has already loaded the full PaySim CSV, replaying 100 rows should add zero
new transaction rows, but it should add 100 first-time Kafka ingestion records. That is
the expected result: both ingestion paths have found the same 100 logical transactions.

## Leakage and label boundary

The event contract separates three groups:

- `transaction` contains ordinary transaction fields.
- `analysis_only_balances` contains `oldbalanceOrg`, `newbalanceOrig`,
  `oldbalanceDest`, and `newbalanceDest`.
- `ground_truth` contains `isFraud` and `isFlaggedFraud`.

The four balance fields are simulator artifacts used only for analysis/audit and are
prohibited as predictive features. Fraud labels are retained for later evaluation, but
must never be supplied as prediction inputs. This structure makes those restrictions
explicit for future feature, API, and model work.

## Local setup

1. Copy `.env.example` to `.env` and replace the PostgreSQL password placeholder.
2. Install Python dependencies with `python -m pip install -r requirements.txt`.
3. Start PostgreSQL and Kafka:

   ```powershell
   docker compose up --detach --wait postgres kafka
   docker compose run --rm kafka-init
   ```

4. Apply the repeatable audit-table migration. This is required once for a PostgreSQL
   volume created before Phase 4; fresh volumes automatically run all three current
   initialization scripts for transactions, ingestion auditing, and point-in-time
   scoring:

   ```powershell
   docker compose exec postgres sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /docker-entrypoint-initdb.d/002_create_transaction_ingestions.sql'
   ```

5. Export the PostgreSQL values from your private `.env` into the terminal session used
   by Python. Docker Compose reads `.env` itself, but Python intentionally reads settings
   from environment variables and never reads or logs the credentials file.

The raw CSV stays under `data/raw/`. It is read from the host and is not copied into a
Docker image or committed to Git.

## Run a replay

Start the consumer in one terminal:

```powershell
python -m src.streaming.consumer --max-messages 100
```

Start the producer in another terminal:

```powershell
python -m src.streaming.producer --start-row 1 --limit 100 --rate 10
```

The producer exits successfully only after the broker acknowledges exactly the requested
number of events. `--rate 10` means approximately 10 events per second. Change the CLI
value or `PAYSIM_REPLAY_RATE` without changing code. `--start-row` and `--limit` select a
repeatable source window while preserving original row numbers and identities.

## Validation, failures, and acknowledgement

The consumer uses manual Kafka acknowledgements:

- A valid event is acknowledged only after its PostgreSQL transaction commits.
- An invalid event is not stored. Its topic position and validation reason are logged
  without logging the rejected payload, then it is acknowledged so it cannot block the
  partition forever.
- A temporary database failure is retried with bounded backoff. If all attempts fail,
  the event remains unacknowledged and is available for redelivery after recovery.
- `Ctrl+C` finishes the current safe outcome and closes the consumer cleanly.

Useful reconciliation queries:

```sql
SELECT COUNT(*)
FROM raw.transaction_ingestions;

SELECT COUNT(DISTINCT ingestion.transaction_id)
FROM raw.transaction_ingestions AS ingestion;

SELECT COUNT(*)
FROM raw.transactions
WHERE first_ingestion_source = 'kafka';
```

When the full Phase 3 batch load already exists, the last count can remain zero after a
replay because the existing transaction correctly keeps `batch` as its first source.

## Tests

Run unit tests with:

```powershell
python -m pytest -m "not integration"
```

PostgreSQL tests require `RUN_POSTGRES_TESTS=1`. The Kafka end-to-end test additionally
requires `RUN_KAFKA_TESTS=1`, a ready broker, the audit migration, and the PostgreSQL
environment variables. GitHub Actions provisions both services and runs the complete
suite automatically for pull requests targeting `main` and pushes to `main`.
