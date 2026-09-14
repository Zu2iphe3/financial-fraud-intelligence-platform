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

## Automated Python tests

On Windows, keep pytest's temporary fixture directories inside the repository to avoid permission problems with the system temporary directory:

```powershell
python -m pytest --basetemp=.pytest_tmp/pytest
```

The `.pytest_tmp/` directory is ignored by Git. This changes only where pytest stores temporary test files; it does not change which tests run or application behavior.

Pull requests targeting `main` and pushes to `main` automatically run the complete pytest suite through GitHub Actions. This checks proposed changes before merge and verifies the resulting `main` branch after a merge or direct push. The workflow creates a clean Ubuntu environment with Python 3.12, temporary PostgreSQL and Apache Kafka services, restores cached pip downloads where available, installs the dependencies declared in `requirements.txt`, initializes the database schemas and Kafka topic, and runs `python -m pytest`.

The result appears in the pull request checks as `pytest (Python 3.12)`. A passing suite produces a green check; a failed test produces a failed check whose workflow log identifies the failing test. This initial workflow performs testing only and does not deploy the project.

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
