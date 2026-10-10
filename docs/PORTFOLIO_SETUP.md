# Portfolio setup and local runbook

This is the authoritative clean-machine sequence for the completed Financial Fraud
Intelligence Platform. Component documents provide deeper troubleshooting and design
detail; this runbook establishes the supported order and the boundary between required
runtime state and optional experiment reproduction.

Commands target Windows PowerShell from the repository root.

## Prerequisites and tested environment

Install:

- Git;
- Python 3.12;
- Docker Desktop with Docker Compose v2;
- Ollama with the `llama3.2:3b` model;
- a Kaggle account/browser session for obtaining PaySim1.

The release candidate was validated with Python `3.12.3`, Docker Compose `5.0.2`,
dbt Core `1.12.2` with `dbt-postgres` `1.11.0`, and Ollama client `0.18.0`.
Compose pins PostgreSQL `18.4-bookworm`, Kafka `4.1.0`, and Airflow
`3.3.1-python3.12`. The approved FastAPI image pins the model-runtime packages recorded
in its manifest.

Plan for at least 16 GB RAM and 15 GB free disk space. The raw CSV is approximately
471 MiB; generated features, a Python environment, Docker images/volumes, Ollama, and
the embedding cache require several additional gigabytes. Full ingestion, dbt builds,
model experiments, and first-time image/model downloads can take substantially longer
than the lightweight API and unit-test paths.

Default ports are PostgreSQL `5432`, Kafka `9092`, FastAPI `8000`, Airflow `8080`,
Streamlit `8501`, and Ollama `11434`. Resolve port conflicts before starting services.

## 1. Clone and install Python dependencies

```powershell
git clone https://github.com/Zu2iphe3/financial-fraud-intelligence-platform.git
cd financial-fraud-intelligence-platform
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import streamlit; print(f'Streamlit {streamlit.__version__} is installed')"
```

The import check must succeed in the same activated environment that will launch the
dashboard. A previously created environment elsewhere on the machine is not required.

Long OneDrive paths can exceed legacy Windows package limits. If installation fails in
a deeply nested checkout, remove only the incomplete new environment and create a fresh
one at a short local path. For example:

```powershell
$projectEnvironment = Join-Path $env:LOCALAPPDATA "ffip-venv"
py -3.12 -m venv $projectEnvironment
& "$projectEnvironment\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import streamlit; print(f'Streamlit {streamlit.__version__} is installed')"
```

This is still a clean project environment created from `requirements.txt`; it is not a
dependency on an undocumented pre-existing dashboard environment.

## 2. Configure local-only environment values

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace every `replace_with_...` placeholder. Use separate local passwords for the
fraud database and Airflow metadata database, and set one random URL-safe
`AIRFLOW_API_JWT_SECRET` of at least 64 characters. Never commit `.env`.

Docker Compose reads `.env` automatically. Python, dbt, and Streamlit read the current
process environment, so load the same values into each new PowerShell session. This
loader ignores comments and blank lines and splits only on the first `=` so URL and
password values are preserved:

```powershell
Get-Content .env |
  Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' } |
  ForEach-Object {
    $parts = $_ -split '=', 2
    Set-Item -Path "Env:$($parts[0])" -Value $parts[1]
  }
```

Validate the complete Compose configuration without starting services:

```powershell
docker compose --profile airflow --profile api config --quiet
```

## 3. Obtain PaySim without redistributing it

Download PaySim1 from the
[original Kaggle listing](https://www.kaggle.com/datasets/ealaxi/paysim1), extract it,
and place exactly one CSV under `data/raw/`. Do not commit the CSV. Attribution and
derived-model terms are documented in
[Dataset and model attribution](DATASET_AND_MODEL_ATTRIBUTION.md).

The supported complete file contains `6,362,620` rows, including `8,213` fraud labels
and `16` flagged-fraud labels.

## 4. Start PostgreSQL and load the historical dataset

```powershell
docker compose up --detach --wait postgres
python -m src.data.paysim_ingestion
```

An empty PostgreSQL volume automatically runs all three scripts in `database/init/`:
the transaction table, Kafka ingestion-audit table, and point-in-time history index.
The loader is chunked and idempotent; rerunning the same source skips existing logical
transactions rather than duplicating them.

Verify reconciliation:

```powershell
docker compose exec postgres psql `
  -U $env:POSTGRES_USER `
  -d $env:POSTGRES_DB `
  -c "SELECT COUNT(*) AS rows, COUNT(*) FILTER (WHERE is_fraud) AS fraud, COUNT(*) FILTER (WHERE is_flagged_fraud) AS flagged FROM raw.transactions;"
```

Expected values are `6,362,620`, `8,213`, and `16`.

Detailed setup: [PostgreSQL](POSTGRES_SETUP.md) and
[batch ingestion](PAYSIM_INGESTION.md).

## 5. Build dbt analytics

```powershell
dbt debug --project-dir dbt --profiles-dir dbt
dbt build --project-dir dbt --profiles-dir dbt
dbt docs generate --project-dir dbt --profiles-dir dbt
```

The Streamlit analytics page reads the trusted `analytics_marts` relations produced by
this build. See [dbt fraud analytics](DBT_ANALYTICS.md) for model grain, tests, lineage,
and documentation commands.

## 6. Start and review Airflow orchestration

Airflow is the batch orchestrator. It does not run Kafka, model training, RAG, or the
application services.

```powershell
docker compose --profile airflow build airflow-init
docker compose --profile airflow up airflow-init
docker compose --profile airflow up --detach `
  postgres airflow-postgres airflow-api-server airflow-scheduler airflow-dag-processor
```

Open <http://localhost:8080>, sign in with the local Airflow values from `.env`, and
trigger `paysim_batch_analytics`. Because the historical load is idempotent, a completed
direct load can be safely reconciled and skipped before dbt runs again.

Detailed retry, failure-blocking and rerun guidance:
[Airflow batch orchestration](AIRFLOW_RUNBOOK.md).

## 7. Validate the separate Kafka replay path

Kafka demonstrates replayable event ingestion; it does not invoke the fraud model.

```powershell
docker compose up --detach --wait kafka
docker compose run --rm kafka-init
```

Start the consumer in one PowerShell window:

```powershell
python -m src.streaming.consumer --max-messages 100
```

Start the producer in another:

```powershell
python -m src.streaming.producer --start-row 1 --limit 100 --rate 10
```

If the full batch load already exists, the replay adds no duplicate transaction rows
but records first-time Kafka delivery metadata. See
[Kafka transaction replay](KAFKA_STREAMING.md) for reconciliation queries.

## 8. Start the approved FastAPI scoring service

The approved `paysim_fraud_model_v2_tuned_review` bundle is versioned under
`artifacts/models/paysim_fraud_model_v2_scoring/`. Normal inference does not need the
historical local MLflow database. Compose mounts the bundle read-only and the loader
verifies its SHA-256 before deserialization.

```powershell
docker compose --profile api build fraud-api
docker compose --profile api up --detach --wait fraud-api
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Submit the repository's safe model-ready fixture:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/predictions `
  -ContentType application/json `
  -InFile tests/fixtures/model_ready_transaction.json
```

Readiness must report `prediction_available: true` and model version
`paysim_fraud_model_v2_tuned_review`. See [FastAPI scoring](PREDICTION_API.md) for the
contract and error behaviour.

## 9. Prepare local RAG and Ollama

Start Ollama if the desktop service is not already running, then install the configured
model and build the deterministic local Chroma index:

```powershell
ollama pull llama3.2:3b
ollama list
python -m src.rag.cli index
python -m src.rag.cli ask "What should an analyst investigate when a transfer goes to a new destination?"
```

The first index build downloads the configured sentence-transformer model. Generated
embedding caches and `chroma_db/` remain local and ignored. Returned sources come from
retrieval metadata, not language-model prose. See [Local RAG assistant](RAG_ASSISTANT.md).

## 10. Start Streamlit

Ensure PostgreSQL/dbt, FastAPI, the Chroma index and Ollama are ready, then run:

```powershell
$streamlitPort = if ([string]::IsNullOrWhiteSpace($env:STREAMLIT_PORT)) {
  8501
} else {
  $env:STREAMLIT_PORT
}

python -m streamlit run src/dashboard/app.py `
  --server.port $streamlitPort
```

The fallback prevents an empty `--server.port` argument and uses Streamlit's documented
project default of `8501`. Open <http://localhost:8501> (or the configured port). Confirm
all four service indicators become available,
then review:

1. analytics KPIs and filtering;
2. a real PostgreSQL transaction through point-in-time risk scoring;
3. a supported policy question with verified sources;
4. an unsupported question returning insufficient evidence.

The experiences fail independently: one unavailable dependency must not disable the
other dashboard capabilities. See [Streamlit dashboard](STREAMLIT_DASHBOARD.md).

A clean checkout does not include generated Chroma state. Run the indexing command in
step 9 once before expecting Policy Search to report ready. The approved scoring bundle
is versioned in the repository, so dashboard/FastAPI inference does not require local
MLflow history or model retraining.

## 11. MLflow provenance and model reproduction

MLflow is an offline experiment/model-selection record, not a runtime API dependency.
The historical local MLflow database is intentionally not distributed because it is
generated state and can contain machine-local artifact URIs. The approved bundle's
manifest preserves the selected historical run ID and exact model/feature identity.

A clean clone can regenerate the chronological datasets and rerun controlled experiments
from PostgreSQL, but those reruns create new MLflow run IDs. Follow
[Behavioural modelling dataset](MODELLING_DATASET_V2.md) and
[Model training and evaluation](MODEL_TRAINING.md) when intentionally reproducing this
resource-intensive workflow.

After local experiments create `mlruns/mlflow.db`, inspect them with:

```powershell
$mlflowDatabase = (Resolve-Path "mlruns/mlflow.db").Path.Replace('\', '/')
mlflow ui --backend-store-uri "sqlite:///$mlflowDatabase"
```

Open <http://localhost:5000>. Exact historical re-promotion requires the original
maintainer's reviewed local run; it is not required to run the versioned scoring API.

## 12. Smoke checks

Run the complete automated suite:

```powershell
python -m pytest --basetemp=.pytest_tmp/release
```

Before release approval, manually confirm:

- PostgreSQL counts reconcile;
- Kafka delivery-audit counts match the controlled replay;
- the Airflow DAG succeeds through dbt;
- dashboard analytics query the marts;
- API readiness and prediction succeed with the approved model;
- policy retrieval answers supported questions with verified sources and declines
  unsupported questions;
- stopping one optional dependency affects only its dashboard capability.

The dated final release-validation record will be added only after this manual sequence
has been reviewed; this runbook does not claim that record prematurely.

## Safe shutdown

Stop Streamlit and any foreground producer/consumer with `Ctrl+C`. Preserve database,
Kafka and Airflow state while stopping only the relevant services:

```powershell
docker compose stop fraud-api
docker compose stop kafka
docker compose --profile airflow stop `
  airflow-dag-processor airflow-scheduler airflow-api-server airflow-postgres
docker compose stop postgres
```

Exit or stop Ollama separately through its local application when desired. These
commands retain named volumes. Do not use `docker compose down --volumes` unless you
explicitly intend to delete local transactions, Kafka data and Airflow history.
