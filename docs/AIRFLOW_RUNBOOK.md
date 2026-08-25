# Airflow batch orchestration runbook

## Purpose

Apache Airflow coordinates the existing historical PaySim pipeline in four visible,
recoverable steps:

```text
check_postgres_ready
        -> ingest_paysim_task
        -> validate_ingestion_task
        -> run_dbt_build
```

The DAG reuses the Phase 3 batch loader and Phase 5 dbt project. Kafka is intentionally
not started, consumed, or produced by this DAG; real-time delivery remains a separate
Phase 4 workflow.

The post-ingestion task is a safety gate. If expected PaySim totals, ingestion outcomes,
or transaction identities do not reconcile, it fails and Airflow blocks the downstream
dbt task. The four PaySim balance fields remain analysis-only leakage fields; orchestration
does not make them safe predictive features.

## One-time local setup

1. From the repository root, create the local environment file if it does not already
   exist:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Keep the existing PostgreSQL values that work for the batch ingestion.
3. Replace `AIRFLOW_DB_PASSWORD` with a local-only, URL-safe password for the Airflow
   metadata database.
4. Replace `AIRFLOW_ADMIN_PASSWORD` with a separate local-only password for signing in
   to the Airflow UI.
5. Replace `AIRFLOW_API_JWT_SECRET` with one random local value of at least 64 characters.
   Every Airflow service must receive the same value so task execution can be authenticated.
6. Never commit or share the populated `.env` file. The values in `.env.example` are
   documentation placeholders and must not be used as real credentials.
7. Confirm that one PaySim CSV exists under `data/raw/`.
8. Ensure Docker Desktop is running.

The real `.env`, raw CSV, Airflow logs, and both PostgreSQL data volumes are excluded
from Git. Airflow uses a dedicated metadata database so DAG history does not mix with
fraud transaction data.

## Start Airflow without Kafka

Build and initialize the local services:

```powershell
docker compose --profile airflow build airflow-init
docker compose --profile airflow up airflow-init
```

Start only the services required by the batch DAG:

```powershell
docker compose --profile airflow up -d postgres airflow-postgres airflow-api-server airflow-scheduler airflow-dag-processor
```

Open <http://localhost:8080> and sign in with `AIRFLOW_ADMIN_USERNAME` and
`AIRFLOW_ADMIN_PASSWORD` from the local `.env` file.

Check that the DAG parsed successfully:

```powershell
docker compose --profile airflow exec airflow-api-server airflow dags list
docker compose --profile airflow exec airflow-api-server airflow dags list-import-errors
```

`paysim_batch_analytics` should appear and the import-error command should show no errors.

## Trigger and observe a run

Use the play button beside `paysim_batch_analytics` in the Airflow UI, or run:

```powershell
docker compose --profile airflow exec airflow-api-server airflow dags trigger paysim_batch_analytics
```

In **Grid** view, select each task to inspect its state, attempt history, and logs. A
successful run proceeds from database readiness through ingestion and validation to dbt.
Logs show summary counts and dataset identity, never transaction rows or credentials.

The first complete ingestion can take time because the loader fingerprints and processes
more than 6.3 million rows. A later run reads the source again but inserts no duplicate
logical transactions: existing rows are reported as `skipped`, validation still passes,
and dbt rebuilds the same analytics results.

## Validate retry behavior

To observe a safe, recoverable readiness retry:

1. Stop only the fraud PostgreSQL service: `docker compose stop postgres`.
2. Trigger the DAG.
3. Open `check_postgres_ready` in Grid view. Its first attempt fails and enters
   **up_for_retry**; the task has three retries separated by 30 seconds.
4. Start the database again: `docker compose start postgres`.
5. The next attempt succeeds and the remaining tasks continue.

The task-instance history and logs retain each attempt, making the recovery visible.

## Failure blocking and recovery

If `validate_ingestion_task` detects a count mismatch, failed record, or duplicate identity,
it fails without retrying. `run_dbt_build` remains **upstream_failed** and cannot publish
analytics from an unsafe load.

After correcting the source, database, or environment configuration:

1. Open the failed DAG run in Grid view.
2. Select only the failed task.
3. Choose **Clear task** and confirm without selecting upstream tasks.
4. Airflow resumes at that boundary; already-successful upstream tasks are not rerun.

For an environment error that affected ingestion, clear `ingest_paysim_task`; validation
and dbt will follow after it succeeds. Do not clear readiness or ingestion when only dbt
configuration failed.

## Rerun and idempotency checks

After two successful runs, verify that logical PaySim transactions remain unique:

```powershell
docker compose exec postgres psql -U $env:POSTGRES_USER -d $env:POSTGRES_DB -c "SELECT COUNT(*) AS rows, COUNT(DISTINCT transaction_id) AS unique_ids FROM raw.transactions;"
```

For the complete documented PaySim file, both values should be `6362620`. The Airflow
validation task also checks the documented fraud total (`8213`), flagged-fraud total
(`16`), and duplicate identity groups before every dbt build.

## Stop services

Stop the batch orchestration services while preserving data and run history:

```powershell
docker compose --profile airflow stop airflow-dag-processor airflow-scheduler airflow-api-server airflow-postgres postgres
```

Do not add `--volumes` unless permanent removal of local Airflow history and database
state is explicitly intended.
