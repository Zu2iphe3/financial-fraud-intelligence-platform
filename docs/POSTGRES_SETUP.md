# Local PostgreSQL transaction storage

Phase 3 introduces PostgreSQL as the durable transaction store shared by the planned batch and Kafka ingestion paths. Docker Compose provides the same database version, health check, initialization, and persistent storage on every development machine.

## Prerequisites

- Docker Desktop with Docker Compose
- Python dependencies installed from `requirements.txt`

## Configure local credentials

Copy `.env.example` to `.env`, then replace `POSTGRES_PASSWORD` with your own local password. The `.env` file is ignored by Git and must never be committed.

Required settings:

| Setting | Purpose |
|---|---|
| `POSTGRES_DB` | Local database name |
| `POSTGRES_USER` | Local database user |
| `POSTGRES_PASSWORD` | Local database password |
| `POSTGRES_HOST` | Host used by Python clients |
| `POSTGRES_PORT` | Host port exposed by Docker Compose |

The values in `.env.example` are placeholders for local development, not operational credentials.

## Start and verify PostgreSQL

From the repository root:

```powershell
docker compose up -d postgres
docker compose ps
```

Wait until the service state is `healthy`, then verify the connection:

```powershell
docker compose exec postgres psql -U fraud_user -d fraud_platform -c "SELECT 1;"
```

Use the database name and user chosen in your `.env` if you changed the example values.

The named volume `financial-fraud-platform-postgres-data` preserves database files when the container stops, restarts, or is recreated. Running `docker compose down` keeps the volume. Running `docker compose down -v` intentionally deletes it and all locally stored records.

## Initialize or reapply the schema

On the first start of an empty volume, the official PostgreSQL image automatically runs `database/init/001_create_raw_transactions.sql`. The script uses safe `IF NOT EXISTS` statements and never drops, truncates, or deletes data.

To reapply it to an existing database:

```powershell
Get-Content database/init/001_create_raw_transactions.sql | docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U fraud_user -d fraud_platform
```

Reapplying the script is safe: existing records remain and schema objects are not duplicated.

Point-in-time model scoring also uses an idempotent destination-history index. Empty
volumes create it automatically. For a database volume initialized before that index
was added, apply it once using the container's existing database configuration:

```powershell
Get-Content -Raw database/init/003_create_point_in_time_feature_index.sql |
  docker compose exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

## PaySim field mapping

The `raw.transactions` table preserves all 11 PaySim source fields. Platform-generated values have separate, clearly named columns.

| PaySim CSV field | PostgreSQL column | PostgreSQL type |
|---|---|---|
| `step` | `step` | `INTEGER` |
| `type` | `transaction_type` | `VARCHAR(16)` |
| `amount` | `amount` | `NUMERIC(20, 2)` |
| `nameOrig` | `name_orig` | `TEXT` |
| `oldbalanceOrg` | `old_balance_orig` | `NUMERIC(20, 2)` |
| `newbalanceOrig` | `new_balance_orig` | `NUMERIC(20, 2)` |
| `nameDest` | `name_dest` | `TEXT` |
| `oldbalanceDest` | `old_balance_dest` | `NUMERIC(20, 2)` |
| `newbalanceDest` | `new_balance_dest` | `NUMERIC(20, 2)` |
| `isFraud` | `is_fraud` | `SMALLINT`, restricted to 0 or 1 |
| `isFlaggedFraud` | `is_flagged_fraud` | `SMALLINT`, restricted to 0 or 1 |

The four balance fields are retained for audit and leakage analysis only. Because PaySim cancels fraudulent transactions, these simulator-generated balances leak information about the fraud label. **They must not be used as predictive features.**

## Logical transaction identity

PaySim does not provide a transaction ID. The platform therefore identifies a source record using:

1. `source_dataset_id`: the lowercase SHA-256 fingerprint of the exact CSV bytes.
2. `source_row_number`: the original one-based data-row number, excluding the header.
3. `transaction_id`: a UUIDv5 calculated from those two values using the fixed project namespace `7d34554e-c520-5df0-a40c-4f7bc686010c`.

Batch ingestion and later Kafka replay must carry the original dataset fingerprint and row number. The same logical row then receives the same `transaction_id` through either path, and database uniqueness prevents it from being stored twice. Separate identical-valued rows remain distinct because their original row numbers differ.

`record_id` is only PostgreSQL's internal generated key. It is deliberately separate from the stable logical identity.

Additional metadata records the first ingestion source (`batch` or `kafka`), optional source filename, and ingestion timestamp. Detailed Kafka delivery and replay auditing is intentionally deferred to Phase 4.

## Run tests

Unit tests run without PostgreSQL:

```powershell
python -m pytest
```

To include database integration tests, start PostgreSQL, set `RUN_POSTGRES_TESTS=1` in the current shell, export the five `POSTGRES_*` values from your `.env`, and run the same command. GitHub Actions automatically creates a temporary PostgreSQL service, initializes the schema, and runs the complete suite for every pull request to `main` and push to `main`.

The integration suite verifies connectivity, all source and metadata columns, exact decimal storage, binary-label rejection, logical-identity uniqueness, expected indexes, and safe schema reapplication.
