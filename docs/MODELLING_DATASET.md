# Leakage-safe PaySim modelling dataset

## Purpose

PH7-01 creates the reproducible input dataset for later fraud-model training. It does
not train, select, or evaluate a model. The pipeline reads the trusted Phase 5
`analytics_marts.fct_transactions` relation in bounded chunks and separates each row
into:

- predictive features;
- the ground-truth target;
- audit and evaluation metadata.

The committed contract at `config/features/paysim_v1.json` is the source of truth for
feature order, data types, source dependencies, time partitions, and prohibited inputs.
Changing that contract requires a new reviewed contract version rather than silently
changing the meaning of an existing modelling dataset.

## Approved predictive features

The `paysim_features_v1` matrix contains exactly six columns in this order:

| Feature | Source | Processing | Reason |
|---|---|---|---|
| `amount_log1p` | `amount` | `log(1 + amount)`, stored as `float32` | Limits the influence of PaySim's extreme amount tail without learning statistics from validation or test periods. |
| `transaction_type_cash_in` | `transaction_type` | one-hot `int8` | Stable representation of `CASH_IN`. |
| `transaction_type_cash_out` | `transaction_type` | one-hot `int8` | Stable representation of `CASH_OUT`. |
| `transaction_type_debit` | `transaction_type` | one-hot `int8` | Stable representation of `DEBIT`. |
| `transaction_type_payment` | `transaction_type` | one-hot `int8` | Stable representation of `PAYMENT`. |
| `transaction_type_transfer` | `transaction_type` | one-hot `int8` | Stable representation of `TRANSFER`. |

This preprocessing is deterministic and is implemented by the same
`transform_features` function that future training and FastAPI inference must call.
Unsupported transaction types, negative/non-finite amounts, invalid labels, or invalid
time values fail before an output dataset is accepted.

## Leakage and restricted fields

The feature contract is an allowlist, not a list of columns to drop after loading. The
database query therefore selects only the columns required to create the features,
target, partitions, and audit metadata.

The following are never predictive inputs:

- `old_balance_orig`, `new_balance_orig`, `old_balance_dest`, and `new_balance_dest`;
- every calculation or renamed feature derived from those four balance fields;
- `is_fraud`, except as the separate target;
- `is_flagged_fraud` and its boolean form because they are existing rule outputs;
- `name_orig` and `name_dest` because raw identifiers encourage memorisation;
- logical transaction identity and ingestion metadata;
- `step`, `simulation_day`, and `simulation_hour_of_day`.

Every configured feature declares its source columns. Contract validation identifies
the proposed feature and fails if any declared dependency is prohibited, even when the
output feature has an innocent or unrelated name. The balance fields remain analysis-
only PaySim simulator artifacts and must not enter future training or prediction code.

## Target and metadata separation

`is_fraud` is written only under each partition's `target/` directory. It never appears
in `features/` or `metadata/`.

Metadata retains `transaction_id`, `source_dataset_id`, `source_row_number`, `step`, and
`simulation_day` for reconciliation and evaluation. It also adds `partition` and
`evaluation_cohort`. Metadata must be joined to predictions only after inference; it is
not a model input.

## Chronological partitions

Random splitting is prohibited because it would mix earlier and later simulator
periods and could allow repeated patterns to appear on both sides of an evaluation.

| Partition | Simulation days | Inclusive steps | Intended use |
|---|---:|---:|---|
| Train | 1–18 | 1–432 | Fit future preprocessing state and models. |
| Validation | 19–24 | 433–576 | Tune and compare future models. |
| Test | 25–31 | 577–743 | Final untouched evaluation period. |

The ranges are contiguous, non-overlapping, and ordered. The generator fails if the
trusted source does not cover the configured ranges or if a row cannot be assigned
exactly once.

These boundaries were selected after comparing time-aligned and row-balanced options
against the complete PaySim file. A row-balanced split would cut through simulated days
14 and 16, or leave only two complete validation days after rounding. The selected
whole-day split retains 6,008,416 training rows and 4,857 fraud examples while giving
validation six days with 1,516 fraud examples and test seven days with 1,840. This
provides broader future-period evaluation without mixing any simulated day across
partitions.

## Final-day simulator artifact

PaySim day 31 contains only 272 transactions and all 272 are labelled fraud. Removing
that period would hide a known data limitation, while using day or step as a feature
would let a model memorise it. The contract therefore:

1. keeps steps 721–743 inside the test partition;
2. labels those rows `final_day_artifact` in evaluation metadata only;
3. labels earlier test rows `standard_period`;
4. prohibits time fields and the cohort label from predictive features.

PH7-02 must report model metrics for the complete test set, the standard-period test
subset, and the final-day cohort separately. PH7-01 supplies those evaluation groups
but deliberately performs no model training or performance reporting.

## Generate the dataset

The local PostgreSQL database must contain the reconciled PaySim data and current dbt
marts. Load the ignored `.env` values into the PowerShell session, start PostgreSQL,
and rebuild the trusted marts if needed:

```powershell
Get-Content .env |
  Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' } |
  ForEach-Object {
    $parts = $_ -split '=', 2
    Set-Item -Path "Env:$($parts[0])" -Value $parts[1]
  }

docker compose up --detach --wait postgres
dbt build --project-dir dbt --profiles-dir dbt
python -m src.features.build_paysim_dataset
```

The generator uses a server-side PostgreSQL cursor and writes 50,000 rows at a time by
default. This prevents the more than 6.3 million source rows from being loaded into
memory together. Use `--chunk-size` only when local memory constraints require a
different bound.

## Generated layout

Generated files are local, reproducible, and excluded from Git:

```text
data/processed/paysim_features/paysim_features_v1/
├── manifest.json
├── train/
│   ├── features/part-*.parquet
│   ├── target/part-*.parquet
│   └── metadata/part-*.parquet
├── validation/
│   ├── features/part-*.parquet
│   ├── target/part-*.parquet
│   └── metadata/part-*.parquet
└── test/
    ├── features/part-*.parquet
    ├── target/part-*.parquet
    └── metadata/part-*.parquet
```

The manifest records the contract version, source dataset fingerprint, exact feature
schema, partition counts and time ranges, fraud counts, and final-day cohort totals.
Generation refuses to overwrite an existing version directory. This protects an
already-reviewed dataset; remove that generated directory deliberately before a full
rebuild.

For the complete documented PaySim dataset, generation must reconcile to:

| Partition | Rows | Fraud rows |
|---|---:|---:|
| Train | 6,008,416 | 4,857 |
| Validation | 192,778 | 1,516 |
| Test | 161,426 | 1,840 |
| **Total** | **6,362,620** | **8,213** |

It must also reconcile 272 final-day rows and 272 final-day fraud labels. Any total or
partition mismatch removes the incomplete output and fails the build.

## Validation

Run unit tests with the Windows-friendly temporary directory:

```powershell
python -m pytest tests/test_feature_contract.py tests/test_feature_pipeline.py --basetemp=.pytest_tmp/ph7-unit
```

With PostgreSQL environment values loaded and the service running, include the chunked
database integration test:

```powershell
$env:RUN_POSTGRES_TESTS = "1"
python -m pytest tests/integration/test_feature_pipeline_integration.py --basetemp=.pytest_tmp/ph7-integration
```

The integration test creates an isolated temporary PostgreSQL schema, validates Parquet
outputs and reconciliation, and removes only that temporary schema afterward.
