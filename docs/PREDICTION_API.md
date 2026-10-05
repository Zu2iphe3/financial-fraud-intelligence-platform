# PaySim fraud model scoring API

PH8-01 exposes the reviewed Phase 7 estimator through FastAPI. It is deliberately a
**model-scoring service**, not an online feature store.

## Service boundary

An upstream feature-preparation component must calculate the complete 20-value
`paysim_features_v2` vector using only information available before the transaction.
The API validates that vector, applies the persisted estimator, and returns a fraud
probability plus a risk classification.

This boundary does not mean a real end user manually calculates behavioural values.
Today, the committed fixture and generated PH7 dataset provide reproducible examples.
A future online feature service can later calculate the same values from transaction
history without changing the model-scoring logic or estimator contract.

The public request never accepts:

- `nameOrig`, `nameDest`, or other raw identifiers;
- PaySim balance or balance-derived fields;
- `isFraud` or `isFlaggedFraud`;
- `step`, simulation day, or Day 31 timing;
- labels, source row numbers, or ingestion metadata.

## Risk classifications

The risk bands in `config/api/fraud_prediction.json` are traceable to PH7-02 validation
review points:

| Classification | Probability | PH7 evidence |
|---|---:|---|
| Low | below `0.1626993418` | Below the reviewed 75%-recall point |
| Moderate | `0.1626993418` to below `0.4920718074` | Between reviewed operating points |
| High | `0.4920718074` or above | At/above the validation maximum-F1 review point |

These are portfolio demonstration bands, not approved production fraud rules. The API
returns a risk label, but it does not claim that any band is a real-world fraud decision.

## Prepare local configuration

From the repository root, create `.env` if it does not already exist:

```powershell
Copy-Item .env.example .env
```

Replace placeholder database and Airflow values with local-only values. Never commit
`.env`; Git ignores it. The API-specific defaults are:

```text
FRAUD_API_PORT=8000
FRAUD_MODEL_ARTIFACT_HOST_PATH=./artifacts/models/paysim_fraud_model_v2_scoring
```

## Promote the reviewed model

The selected estimator is stored in ignored local MLflow evidence. Package it without
retraining, tuning, loading the test partition, or selecting a new threshold:

```powershell
python -m src.models.promote_scoring_model
```

The command verifies the approved MLflow run ID, its historical contract hash, the
`high_child_weight` trial, and `test_partition_loaded=False`. It also proves that the 20
feature names recorded with the estimator match the finalized committed v2 contract.
The deployable artifact records the current contract hash and is written under
`artifacts/models/paysim_fraud_model_v2_scoring/`. Promotion refuses to replace an
existing artifact silently.

After promotion froze the artifact and threshold, the exact scoring bundle was evaluated
once on chronological test days 25-31. This later evaluation did not change promotion,
the model, the feature contract, or either risk boundary. See
[`MODELLING_DATASET_V2.md`](MODELLING_DATASET_V2.md) for the held-out evidence.

The model binary, MLflow database, and generated model state remain outside Git.
The API image pins model-runtime packages to the versions recorded by this artifact,
so a fresh container cannot silently use a different XGBoost or scikit-learn version.

## Run directly with Python

```powershell
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Open interactive OpenAPI documentation at <http://127.0.0.1:8000/docs>.

Check readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Submit the safe model-ready fixture:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/predictions `
  -ContentType application/json `
  -InFile tests/fixtures/model_ready_transaction.json
```

The response contains `fraud_probability`, `risk_classification`, `model_version`,
`risk_policy_version`, and the explicit non-production `risk_policy_status`.

## Run with Docker Compose

Build and start only the API profile:

```powershell
docker compose --profile api build fraud-api
docker compose --profile api up --detach --wait fraud-api
```

Compose mounts the ignored host artifact at `/opt/app/model` read-only. The model is
not copied into the Docker image. Use the same readiness and prediction commands shown
above, then stop the service with:

```powershell
docker compose --profile api down
```

## Endpoints and failures

### `GET /health/ready`

- Returns `200` and `prediction_available: true` when a compatible model is loaded.
- Returns `503` and `prediction_available: false` when the artifact is missing,
  corrupt, or incompatible.

### `POST /v1/predictions`

- Returns `200` for one valid 20-feature vector.
- Returns `422` for missing, extra, non-finite, or domain-invalid fields. The response
  identifies the invalid field and inference does not run.
- Returns `503` when the model is unavailable; no score is fabricated.
- Returns a controlled `500` when validated input reaches an unexpected inference
  failure.

Errors do not expose stack traces, local paths, configuration values, or complete
payloads. Operational logs record only safe event metadata such as the model version,
risk class, validation count, and error type. Responses include `X-Request-ID` for
correlation.

## Validation

Run focused API tests:

```powershell
python -m pytest `
  tests/test_api_config.py `
  tests/test_api_contracts.py `
  tests/test_api_endpoints.py `
  tests/test_scoring_artifact.py `
  tests/test_model_promotion.py `
  tests/integration/test_api_model_integration.py `
  --basetemp=.pytest_tmp/ph8-01
```

Run the complete regression suite:

```powershell
python -m pytest --basetemp=.pytest_tmp/pytest
```

## Future integration

The Streamlit dashboard calls this API rather than importing the estimator. The local
point-in-time adapter below prepares stored historical transactions; extracting it into
a continuously updated online feature service remains future work. Authentication,
prediction persistence, live feature-state management, and cloud deployment remain
outside the current scoring workflow.

## Point-in-time historical transaction scoring

`src.features.point_in_time.PointInTimeFeatureService` reconstructs the reviewed v2
vector for an existing PostgreSQL `transaction_id`. It selects only transaction
identity, PaySim step, type, amount, and origin/destination identifiers; history is
restricted to the same dataset and destination at strictly earlier steps. The service
then reuses `calculate_destination_history()` and `transform_features()` rather than
maintaining a second inference-only feature definition.

`src.dashboard.transaction_scoring.HistoricalTransactionScoringService` joins that
feature preparation to the existing `PredictionClient` and `/v1/predictions` endpoint.
The Fraud Risk Scoring page uses this service to analyse stored transaction IDs without
asking an investigator to prepare model features. Missing transactions, database
failures, or invalid history fail explicitly and never substitute demonstration or
zero-filled data.

The PostgreSQL initialization scripts include a composite dataset/destination/step
index for interactive history lookup. Existing persistent database volumes must apply
`database/init/003_create_point_in_time_feature_index.sql` once; new empty volumes run
it automatically with the other initialization scripts.
