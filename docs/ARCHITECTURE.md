# Detailed implemented architecture

This diagram expands the compact [README architecture](../README.md) into the
implemented data, analytics, machine-learning, runtime-scoring, policy-assistant, and
engineering-validation paths.

Repeated PostgreSQL nodes refer to the same transaction database viewed from different
workflows. The **Approved versioned XGBoost scoring artifact** in ML development and the
**Approved XGBoost artifact from ML development** in runtime scoring are the same
versioned `paysim_fraud_model_v2_scoring` bundle. FastAPI loads that bundle directly;
MLflow is experiment and model-selection provenance, not a runtime dependency.

```mermaid
flowchart TB
    subgraph ROW1[" "]
        direction LR

        subgraph ANALYTICS[2. Analytics]
            direction TB
            ANALYTICS_DB[(PostgreSQL<br/>same transaction store)]
            AIRFLOW[Apache Airflow<br/>batch orchestration]
            BATCH_DAG[Batch ingestion → reconciliation → dbt]
            DBT[dbt staging and transformations]
            MARTS[(Trusted analytics marts)]
            OVERVIEW[Streamlit<br/>Overview & Analytics]

            ANALYTICS_DB --> DBT --> MARTS --> OVERVIEW
            AIRFLOW -. orchestrates .-> BATCH_DAG
            BATCH_DAG -. runs and gates .-> DBT
        end

        subgraph INGEST[1. Data Ingestion]
            direction TB
            CSV[PaySim CSV<br/>local, not committed]
            BATCH[Chunked batch loader]
            PRODUCER[Kafka replay producer]
            KAFKA[Apache Kafka]
            CONSUMER[Validated consumer]
            POSTGRES[(PostgreSQL<br/>transactions)]

            CSV --> BATCH --> POSTGRES
            CSV --> PRODUCER --> KAFKA --> CONSUMER --> POSTGRES
        end
    end

    subgraph ROW2[" "]
        direction LR

        subgraph SCORING[4. Runtime Fraud Scoring]
            direction TB
            SCORING_UI[Streamlit Fraud Risk Scoring<br/>transaction review]
            SCORING_DB[(PostgreSQL<br/>same transaction store)]
            RECONSTRUCT[Point-in-time<br/>feature reconstruction]
            API[FastAPI scoring service]
            RUNTIME_ARTIFACT[Approved XGBoost artifact<br/>from ML development]
            RESULT[Risk result returned to Streamlit<br/>probability, band and model version]
            NO_MLFLOW[MLflow is not a<br/>runtime dependency]

            SCORING_UI --> RECONSTRUCT --> API --> RESULT
            SCORING_DB --> RECONSTRUCT
            RUNTIME_ARTIFACT -->|loaded directly| API
        end

        subgraph ML[3. ML Development]
            direction TB
            ML_DB[(PostgreSQL<br/>same transaction store)]
            FEATURES[Chronological point-in-time<br/>feature dataset]
            DEVELOPMENT[Model development<br/>and XGBoost selection]
            MLFLOW[MLflow<br/>experiment provenance]
            ARTIFACT[Approved versioned<br/>XGBoost scoring artifact]

            ML_DB --> FEATURES --> DEVELOPMENT --> MLFLOW --> ARTIFACT
        end
    end

    subgraph ROW3[" "]
        direction LR

        subgraph ENGINEERING[6. Engineering Validation]
            direction TB
            CI[GitHub Actions CI]
            CHECKS[Python and integration tests<br/>dbt and Compose validation<br/>secret scanning]

            CI --> CHECKS
        end

        subgraph RAG[5. Policy Assistant / RAG]
            direction TB
            POLICIES[Fictional policy documents]
            EMBEDDINGS[Chunking and local embeddings]
            CHROMA[(Chroma vector store)]
            POLICY_UI[Streamlit Policy Assistant<br/>question]
            RETRIEVAL[Semantic retrieval and grounding]
            OLLAMA[Ollama / Llama 3.2]
            GROUNDED[Grounded answer with verified sources<br/>returned to Streamlit Policy Assistant]

            POLICIES --> EMBEDDINGS --> CHROMA --> RETRIEVAL
            POLICY_UI --> RETRIEVAL --> OLLAMA --> GROUNDED
        end
    end

    style ROW1 fill:none,stroke:none
    style ROW2 fill:none,stroke:none
    style ROW3 fill:none,stroke:none

    POSTGRES ~~~ ML_DB
    ANALYTICS_DB ~~~ SCORING_DB
    ARTIFACT ~~~ POLICIES
    RESULT ~~~ CI
```

## System boundaries

- Batch ingestion and Kafka replay both persist the same deterministic PaySim
  transaction identities to PostgreSQL. Kafka does not perform fraud inference.
- Airflow orchestrates the batch ingestion, reconciliation, and dbt workflow only. It
  does not orchestrate Kafka, model training, RAG, or application serving.
- dbt analytics marts serve the Overview & Analytics experience. They are separate
  from the point-in-time behavioural feature reconstruction used for scoring.
- MLflow records offline experiments and selection evidence. Runtime FastAPI scoring
  loads the approved repository-versioned model bundle without connecting to MLflow.
- Fraud Risk Scoring reconstructs leakage-safe features from strictly earlier
  transaction history before calling FastAPI.
- The local policy assistant uses only the fictional policy corpus, Chroma retrieval,
  and Ollama generation. It neither trains nor changes the fraud model.
- GitHub Actions validates repository configuration, contracts, automated tests, and
  secret safety; it is not part of the transaction or inference flow.
