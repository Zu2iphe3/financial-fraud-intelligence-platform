# Financial Fraud Intelligence Platform

An end-to-end data and AI portfolio platform for simulated real-time financial transaction processing, fraud detection, analytics, MLOps, and a retrieval-augmented fraud analyst assistant.

## Business problem

A fictional digital-payments company needs to ingest transaction events reliably, create trusted analytical data, identify suspicious transactions, expose fraud-risk predictions, and help analysts retrieve guidance from fraud-policy documents.

This repository uses public or synthetic data, original code, and fictional policy documents only.

## Planned architecture

```text
PaySim -> Python producer -> Kafka -> Python consumer -> PostgreSQL
                                                        |
                                                        v
                                                dbt transformations
                                                   /           \
                                                  v             v
                                        analytics marts    ML features
                                                  |             |
                                                  v             v
                                           Streamlit       XGBoost
                                                              |
                                                     MLflow -> FastAPI

Fictional fraud documents -> chunks -> embeddings -> vector store
                         -> retrieval -> Ollama -> analyst assistant
```

## Dataset

The project uses the [PaySim synthetic financial dataset](https://www.kaggle.com/datasets/ealaxi/paysim1). PaySim simulates mobile-money transactions and includes labels identifying simulated fraud.

The complete dataset is intentionally excluded from Git. After downloading it, place the CSV file under `data/raw/`. A small permitted sample may be added to `data/sample/` later for tests and demonstrations.

Phase 2 documentation:

- [PaySim data dictionary](docs/DATA_DICTIONARY.md)
- [Exploratory analysis findings](docs/EDA_FINDINGS.md)
- [Reproducible exploration notebook](notebooks/01_paysim_exploration.ipynb)

Phase 3 local database setup:

- [PostgreSQL setup and transaction schema](docs/POSTGRES_SETUP.md)
- [PaySim batch ingestion and reconciliation](docs/PAYSIM_INGESTION.md)

Phase 4 local streaming setup:

- [Kafka transaction replay pipeline](docs/KAFKA_STREAMING.md)

Phase 5 analytics engineering:

- [dbt fraud analytics layer](docs/DBT_ANALYTICS.md)

Phase 6 batch orchestration:

- [Airflow batch pipeline runbook](docs/AIRFLOW_RUNBOOK.md)

Phase 7 machine learning and experiment tracking:

- [Leakage-safe modelling dataset](docs/MODELLING_DATASET.md)
- [Leakage-safe behavioural modelling dataset review](docs/MODELLING_DATASET_V2.md)
- [Fraud model training, selection, and evaluation](docs/MODEL_TRAINING.md)
- [ML fraud detection learning notes](docs/ML_FRAUD_DETECTION_LEARNING_NOTES.md)
- [Non-selected PH7-02 experiment evidence](docs/experiments/PH7_02_EXPERIMENT_EVIDENCE.md)

Phase 8 prediction serving:

- [FastAPI fraud model scoring service](docs/PREDICTION_API.md)

Phase 9 RAG analyst assistant:

- [Fictional fraud-policy knowledge base](docs/FRAUD_POLICY_KNOWLEDGE_BASE.md)
- [Local grounded RAG analyst assistant](docs/RAG_ASSISTANT.md)

Phase 10 portfolio dashboard:

- [Streamlit fraud intelligence dashboard](docs/STREAMLIT_DASHBOARD.md)

## Technology stack

- Python, SQL, Git, and GitHub
- Docker and Docker Compose
- Apache Kafka and PostgreSQL
- dbt Core and Apache Airflow
- pandas, scikit-learn, XGBoost, and MLflow
- FastAPI
- LangChain, local embeddings, a local vector store, and Ollama
- Streamlit
- pytest, dbt tests, and GitHub Actions

## Automated platform validation

On Windows, keep pytest's temporary fixture directories inside the repository to avoid permission problems with the system temporary directory:

```powershell
python -m pytest --basetemp=.pytest_tmp/pytest
```

The `.pytest_tmp/` directory is ignored by Git. This changes only where pytest stores temporary test files; it does not change which tests run or application behavior.

Pull requests targeting `main` and pushes to `main` automatically run the final platform checks through GitHub Actions. This validates proposed changes before merge and verifies the resulting `main` branch after a merge or direct push. The workflow:

- validates the complete Docker Compose configuration, including the optional API and Airflow profiles, without building or starting those heavyweight services;
- restores cached pip downloads where available and installs `requirements.txt` in a clean Python 3.12 environment;
- checks the dbt connection and compiles every model, test, source reference, macro and YAML definition without processing the full PaySim dataset;
- starts temporary PostgreSQL and Apache Kafka services, initializes their schemas/topic and runs the complete pytest collection, including enabled database, Kafka and FastAPI integration tests; and
- scans the complete Git history with Gitleaks, failing on recognized credential patterns while keeping secret values out of comments and uploaded report artifacts.

The pull request displays `Platform tests (Python 3.12)` and `Secret scan (Gitleaks)` checks. Clearly named workflow steps and normal pytest/dbt diagnostics identify the failed test, dbt resource, service configuration or secret location. These checks validate the repository but never deploy it.

GitHub-hosted CI intentionally does not ingest the 6.3-million-row PaySim CSV, execute a full data-populated `dbt build`, retrain models, start Airflow or the Streamlit dashboard, download local model artifacts, or run Ollama and its embedding models. Those data-heavy and local-service workflows remain covered by their documented local validation procedures. FastAPI contracts and model-service behaviour, RAG grounding behaviour and dashboard behaviour still run through the automated Python suite using repository fixtures and mocks where a live local dependency would be disproportionate.

## Build phases

1. Repository and development foundation
2. Dataset acquisition and exploration
3. Local PostgreSQL pipeline
4. Simulated real-time streaming with Kafka
5. Analytics engineering with dbt
6. Workflow orchestration with Airflow
7. Machine learning and experiment tracking
8. Fraud prediction API
9. RAG analyst assistant
10. Dashboard, automated checks, and portfolio release

## Current status

Phase 9 - RAG analyst assistant (in progress).
