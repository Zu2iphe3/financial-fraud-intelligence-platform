# dbt fraud analytics layer

## Purpose

Phase 5 uses dbt Core to turn the validated PostgreSQL transaction source into a
documented, tested analytics layer:

```text
raw.transactions
       |
       v
analytics_staging.stg_transactions
       |
       v
analytics_intermediate.int_transactions_enriched
       |
       v
analytics_marts.fct_transactions
       |-------------------------------|
       v                               v
analytics_marts.mart_fraud_by_day   analytics_marts.mart_fraud_by_transaction_type
```

PostgreSQL remains the only database. dbt creates views in separate schemas so each
layer has one clear responsibility without copying the 6.3-million-row source into a
new warehouse.

## Model responsibilities

### Source

`raw.transactions` is the Phase 3 transaction store. dbt declares and tests it but does
not alter it.

### Staging

`stg_transactions` keeps exactly one row for every source transaction. It explicitly
selects and casts every source column so downstream models do not depend on `select *`
or accidental source-type changes.

### Intermediate

`int_transactions_enriched` preserves transaction grain and adds reusable PaySim time
dimensions:

- `simulation_day = ((step - 1) / 24) + 1`
- `simulation_hour_of_day = mod(step - 1, 24) + 1`

Subtracting one before grouping is important because PaySim steps begin at 1 rather
than 0. The model also provides readable boolean forms of the fraud labels.

### Marts

- `fct_transactions`: one row per `transaction_id` for trusted transaction analytics.
- `mart_fraud_by_day`: one row per `simulation_day` with count, amount, fraud,
  flagged-fraud, fraud-rate, observed-hour, and partial-day measures.
- `mart_fraud_by_transaction_type`: one row per `transaction_type` with stable count,
  amount, fraud, flagged-fraud, and fraud-rate measures.

Fraud rates are fractions from 0 to 1. For example, `0.01` means one percent.

## Leakage and prediction boundary

The following PaySim simulator fields remain visible only for analysis and audit:

- `old_balance_orig`
- `new_balance_orig`
- `old_balance_dest`
- `new_balance_dest`

They are prohibited as predictive features because fraudulent PaySim transactions are
cancelled, which causes their resulting balances to leak the fraud outcome. `is_fraud`,
`is_flagged_fraud`, and their boolean forms are ground truth for monitoring and model
evaluation; they must never be supplied as prediction inputs.

This ticket creates analytics datasets only. It does not create the Phase 7 machine-
learning feature pipeline.

## Configure the connection

Install the repository dependencies and start PostgreSQL:

```powershell
python -m pip install -r requirements.txt
docker compose up --detach --wait postgres
```

On Windows, a deeply nested repository path can exceed the legacy path-length limit
while pip installs dbt dependencies. If that happens, keep the repository where it is
and create the virtual environment at a shorter local path:

```powershell
$dbtEnvironment = Join-Path $env:LOCALAPPDATA "ffip-dbt-venv"
python -m venv $dbtEnvironment
& "$dbtEnvironment\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
```

This environment stays outside Git and contains dependencies only—not project data or
credentials.

The committed `dbt/profiles.yml` contains environment-variable references only. It does
not contain credentials. Load the private values from the ignored `.env` into each
PowerShell session that runs dbt:

```powershell
Get-Content .env |
  Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' } |
  ForEach-Object {
    $parts = $_ -split '=', 2
    Set-Item -Path "Env:$($parts[0])" -Value $parts[1]
  }
```

Never replace the environment references in `dbt/profiles.yml` with a real password.

## Build and test

Run all commands from the repository root:

```powershell
dbt debug --project-dir dbt --profiles-dir dbt
dbt build --project-dir dbt --profiles-dir dbt
```

`dbt build` creates the staging, intermediate, and mart views in dependency order and
runs source, model, relationship, grain, reconciliation, dataset-fact, and final-day
tests. A test returns failing records, so zero returned records means the assertion
passed.

The complete PaySim dataset should reconcile to:

| Measure | Expected result |
|---|---:|
| Transactions | 6,362,620 |
| Fraudulent transactions | 8,213 |
| Flagged transactions | 16 |
| Fraud transaction types | CASH_OUT and TRANSFER only |
| Final simulation day | 31 |
| Final-day hours observed | 23 |
| Final-day transactions | 272 |
| Final-day fraud count | 272 |
| Final-day fraud rate | 1.0 |

Day 31 is a partial simulation day with unusually low volume and a 100% fraud rate. It
is deliberately exposed through `is_partial_day`; it is not filtered or silently
treated as a normal full day.

## Query the marts

```sql
SELECT *
FROM analytics_marts.mart_fraud_by_transaction_type
ORDER BY transaction_type;

SELECT *
FROM analytics_marts.mart_fraud_by_day
ORDER BY simulation_day;

SELECT COUNT(*)
FROM analytics_marts.fct_transactions;
```

Dashboard, orchestration, and model code should consume these documented marts instead
of recreating measures directly from `raw.transactions`.

## Generate and inspect lineage documentation

```powershell
dbt docs generate --project-dir dbt --profiles-dir dbt
dbt docs serve --project-dir dbt --profiles-dir dbt
```

The local documentation site shows sources, models, columns, tests, and lineage. Stop
the local documentation server with `Ctrl+C`. Generated `dbt/target/`, `dbt/logs/`, and
`dbt/dbt_packages/` content is ignored and must not be committed.

When finished, stop PostgreSQL without deleting its persistent volume:

```powershell
docker compose stop postgres
```
