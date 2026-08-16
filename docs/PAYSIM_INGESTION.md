# PaySim batch ingestion

PH3-02 loads the complete local PaySim CSV into the `raw.transactions` PostgreSQL table established in PH3-01. The loader validates the source, processes it in bounded chunks, preserves exact source values, prevents duplicate logical transactions, and reconciles the final database totals.

## Safety and modelling boundaries

- The raw CSV stays under `data/raw/` and is ignored by Git.
- The CSV is never copied into a Docker image or generated artifact.
- Database credentials come from the five `POSTGRES_*` environment variables.
- Logs contain filenames, dataset identifiers, chunk numbers, counts, and safe error details—not credentials or complete transaction rows.
- The four PaySim balance fields are stored for audit and analysis only. They contain simulator leakage and **must not be used as predictive features**.

## Prerequisites

1. Follow [the PostgreSQL setup](POSTGRES_SETUP.md) and start the healthy database service.
2. Place exactly one PaySim CSV under `data/raw/`, or select a repository-relative CSV explicitly with `--csv`.
3. Export `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, and `POSTGRES_PORT` in the shell that will run Python.
4. Install the dependencies from `requirements.txt`.

The complete PaySim file expected by this project contains:

| Measure | Expected value |
|---|---:|
| Transactions | 6,362,620 |
| Fraud transactions | 8,213 |
| Flagged-fraud transactions | 16 |

## Run ingestion

From the repository root, let the loader discover the only CSV under `data/raw/`:

```powershell
python -m src.data.paysim_ingestion
```

Select a file or chunk size explicitly when needed:

```powershell
python -m src.data.paysim_ingestion --csv data/raw/PS_20174392719_1491204439457_log.csv --chunk-size 50000
```

The default chunk size is 50,000 rows. It can also be changed without editing code:

```powershell
$env:PAYSIM_INGESTION_CHUNK_SIZE = "25000"
python -m src.data.paysim_ingestion
```

Smaller chunks use less memory and create shorter database transactions, while larger chunks can improve throughput. The complete CSV is never loaded into memory at once.

## Exact monetary values

`amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, and `newbalanceDest` are read directly from CSV text into Python `Decimal` values. They never pass through binary floating-point numbers. PostgreSQL therefore receives exact decimal values for its `NUMERIC(20,2)` columns, including values such as `0.10`.

## Transaction identity and safe reruns

Every row receives the stable identity defined in PH3-01:

```text
UUIDv5(exact CSV SHA-256 fingerprint + original one-based data-row number)
```

The original data-row number excludes the header. PostgreSQL uniqueness constraints protect both the generated `transaction_id` and the dataset-fingerprint/row-number pair.

The repeat-run policy is idempotent:

- rows missing from a partial earlier run are inserted;
- rows already stored are skipped;
- existing source values are not silently overwritten;
- unintended duplicate transactions are not created.

## Chunk transactions and failure recovery

Each validated chunk is copied into a session-local PostgreSQL staging table and moved into `raw.transactions` inside one database transaction. If that database batch fails, the whole chunk is rolled back and counted as failed. It is never reported as inserted.

After correcting the database or source problem, run the same command again. Previously committed rows will be skipped and missing rows can be inserted safely.

## Understand the summary

Progress and final logs keep these outcomes separate:

```text
processed=6,362,620, inserted=6,362,620, skipped=0, failed=0, reconciled=yes
```

- `processed`: records given a final outcome during this run.
- `inserted`: new records committed to PostgreSQL.
- `skipped`: valid records already present under the stable identity.
- `failed`: records in a source-validation or rolled-back database failure.
- `reconciled`: whether known, source, and database totals agree.

The invariant is always:

```text
processed = inserted + skipped + failed
```

A successful first run into an empty table should insert all 6,362,620 records. A complete repeat run should insert zero and skip all 6,362,620 records.

## Reconciliation

After the final chunk, the loader compares:

1. totals read from the source;
2. the known complete PaySim totals above;
3. PostgreSQL totals filtered to the exact dataset fingerprint.

Transaction, fraud, or flagged-fraud differences cause a non-zero command exit rather than a false success.

You can also inspect the database directly:

```sql
SELECT
    COUNT(*) AS transactions,
    SUM(is_fraud) AS fraud,
    SUM(is_flagged_fraud) AS flagged_fraud
FROM raw.transactions;
```

## Tests

Run unit tests without PostgreSQL:

```powershell
python -m pytest
```

To include integration tests locally, start PostgreSQL, export the five `POSTGRES_*` settings, set `RUN_POSTGRES_TESTS=1`, and run the same command. GitHub Actions supplies a temporary PostgreSQL service and runs the complete suite automatically for pull requests targeting `main` and pushes to `main`.

The test suite covers source discovery, schema and value validation, chunk boundaries, exact decimals, stable identities, multi-chunk persistence, safe repeat runs, rollback behavior, failure summaries, and reconciliation.
