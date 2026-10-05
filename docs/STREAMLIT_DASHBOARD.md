# Streamlit fraud intelligence dashboard

## Purpose and boundaries

PH10-01 provides the portfolio interface for three completed platform capabilities:

| Dashboard experience | Authoritative backend | Meaning |
|---|---|---|
| Overview & Analytics | PostgreSQL `analytics_marts` dbt models | Historical simulated activity |
| Fraud Risk Scoring | PH8 FastAPI service | Model-estimated fraud risk |
| Policy Assistant | PH9 local Chroma retrieval and Ollama | Fictional policy guidance |

These experiences are deliberately separate. Historical fraud labels are not a live
prediction, a model probability is not proof of fraud, and fictional policy guidance
is not legal or operational advice.

The application is a local learning system. It does not add production authentication,
a continuously updated online feature store, deployment, or automated decision-making.

## Interface structure

Run one Streamlit process and use its sidebar navigation:

```text
Overview & Analytics
Fraud Risk Scoring
Policy Assistant
```

The Overview page reports PostgreSQL analytics, FastAPI prediction, local policy
search, and Ollama readiness in one compact horizontal service strip. The sidebar
shows only the overall available-service count so navigation stays concise. A failed
dependency disables only the page capability that needs it.

## Configure the local environment

Create `.env` from the safe example if needed and replace local placeholders. The real
file is ignored and must never be committed.

```powershell
Copy-Item .env.example .env
```

Load `.env` into the PowerShell session that will start services and Streamlit:

```powershell
Get-Content .env |
  Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' } |
  ForEach-Object {
    $parts = $_ -split '=', 2
    Set-Item -Path "Env:$($parts[0])" -Value $parts[1]
  }
```

Dashboard-specific defaults are:

```text
FRAUD_API_BASE_URL=http://127.0.0.1:8000
STREAMLIT_PORT=8501
DASHBOARD_REQUEST_TIMEOUT_SECONDS=5
DASHBOARD_HEALTH_CACHE_SECONDS=20
```

Install declared dependencies in the project virtual environment:

```powershell
python -m pip install -r requirements.txt
```

## Prepare historical analytics

Start PostgreSQL and ensure the PaySim data has been ingested, then build the dbt
analytics layer:

```powershell
docker compose up --detach --wait postgres
dbt build --project-dir dbt --profiles-dir dbt
```

The dashboard reads only these trusted marts:

- `analytics_marts.fct_transactions`;
- `analytics_marts.mart_fraud_by_day`;
- `analytics_marts.mart_fraud_by_transaction_type`.

It never reads the raw CSV or `raw.transactions`. Filters generate parameterized,
read-only `SELECT` queries and do not alter source data.

## Prepare the prediction service

If the reviewed scoring artifact does not exist locally, promote it using the PH8
runbook. Promotion packages the reviewed model without retraining it:

```powershell
python -m src.models.promote_scoring_model
```

Start FastAPI directly:

```powershell
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Alternatively, use the documented API Docker profile. Confirm readiness at
<http://127.0.0.1:8000/health/ready>.

The scoring page accepts an existing PostgreSQL `transaction_id`. Its point-in-time
adapter loads the transaction and only earlier activity for the same dataset and
destination. All transactions from the selected transaction's PaySim hour and all
future hours are excluded, matching the reviewed offline behavioural feature semantics.
The adapter reuses that feature implementation and sends the exact 20-feature contract
to FastAPI. The investigator never enters engineered values, balance fields, fraud
labels, flagged-fraud labels, or Day 31 indicators.

## Prepare the policy assistant

Start Ollama and ensure the configured local chat model is present:

```powershell
ollama pull llama3.2:3b
ollama list
```

Build the deterministic local policy index if needed:

```powershell
python -m src.rag.cli index
```

Policy Search health loads the configured embedding model and executes a real
semantic readiness query against Chroma. It reports available only when the same
retrieval path used by a user question succeeds. Keep `HF_HOME` on a durable path;
do not use a temporary directory with `HF_HUB_OFFLINE=1`.

Streamlit reuses the existing PH9 retrieval and answer-generation implementation. It
does not duplicate chunking, relevance, grounding, refusal, prompt-injection, or source
validation logic. The natural-language answer and application-controlled sources are
rendered separately.

## Start the dashboard

From the repository root:

```powershell
python -m streamlit run src/dashboard/app.py `
  --server.port $env:STREAMLIT_PORT
```

Open <http://localhost:8501>. Use **Refresh connections** after starting or stopping
a dependency; normal status results are cached briefly to keep widget interactions
responsive.

Streamlit source watching is intentionally disabled. PyTorch and Transformers expose
dynamic modules that produce noisy watcher errors and can leave a development process
with mixed old/new imports after a hot reload. Restart Streamlit after source changes;
normal page navigation and data refreshes do not require a restart.

## Page guide

### Overview & Analytics

- Filter by simulation-day range and transaction type.
- Review transaction volume, fraud count, fraud rate, and fraud amount.
- Explore interactive time and transaction-type views.
- Day 31 remains visible and is identified as the known partial simulator-artifact
  cohort rather than silently treated as a normal day.

### Fraud Risk Scoring

- Load the compact **Transactions for review** list from the trusted PostgreSQL mart,
  or retain direct lookup when a specific transaction ID is known.
- Enter an existing PostgreSQL transaction ID and select **Analyse risk**.
- Review the transaction, simulation time, and readable prior destination activity.
- Submit through the point-in-time adapter and FastAPI; Streamlit never imports the
  estimator or fabricates missing history.
- Review probability, risk classification, model version, and human-review guidance.
- Expand **Technical details** only when the exact read-only 20-feature vector is needed.
- Expand **Model information** for separately labelled validation/model-selection and
  final held-out test evidence. Ranking metrics remain distinct from metrics calculated
  at the unchanged frozen threshold.

### Policy Assistant

- Enter a supported investigation, escalation, or risk-interpretation question.
- Review the grounded natural-language response.
- Expand the fictional document and section sources supplied by retrieval metadata.
- Unsupported questions return the PH9 insufficient-evidence response.

## Failure isolation

The application intentionally catches dependency failures at the affected boundary:

- PostgreSQL unavailable: the analytics page gives setup guidance.
- FastAPI unavailable: scoring shows a controlled availability message.
- Chroma index absent: policy health reports setup required.
- Ollama unavailable: policy answer generation reports unavailable.

No state exposes passwords, local paths, stack traces, complete request payloads, raw
embeddings, or model internals. Other pages remain usable when one service fails.

## Validation

Run focused deterministic tests:

```powershell
python -m pytest `
  tests/test_dashboard_config.py `
  tests/test_dashboard_analytics.py `
  tests/test_dashboard_prediction.py `
  tests/test_dashboard_policy.py `
  tests/test_dashboard_health.py `
  tests/test_dashboard_app.py `
  --basetemp=.pytest_tmp/ph10-dashboard
```

With PostgreSQL marts and FastAPI running, run the opt-in integration check:

```powershell
$env:RUN_DASHBOARD_TESTS="1"
python -m pytest tests/integration/test_dashboard_integration.py `
  --basetemp=.pytest_tmp/ph10-dashboard-live
```

The PH9 real embedding/Ollama integration suite remains the authoritative end-to-end
validation for the reused RAG backend. Run the complete project regression suite with:

```powershell
python -m pytest --basetemp=.pytest_tmp/pytest
```

## Manual acceptance checklist

1. Open all three pages through sidebar navigation.
2. Confirm metrics and charts update for day and transaction-type filters.
3. Confirm Day 31 displays its partial-day warning.
4. Analyse a real transaction with history and one cold-start transaction.
5. Confirm the form accepts only a transaction ID and engineered values are read-only
   inside **Technical details**.
6. Ask one supported and one unsupported fictional policy question.
7. Confirm policy sources are shown separately from the generated answer.
8. Stop PostgreSQL, FastAPI, or Ollama one at a time and refresh connections.
9. Confirm each unavailable service affects only its own capability.
10. Confirm the UI displays no raw JSON, credentials, stack traces, or private paths.
