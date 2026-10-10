# Third-party notices

The repository's [Evaluation License](LICENSE) applies only to original material authored
for the Financial Fraud Intelligence Platform. The following external or derived material
remains governed by its own terms. This summary is provided for attribution and release
clarity; the authoritative license distributed by each provider controls.

## PaySim data and derived scoring model

- **PaySim1 dataset:** Creative Commons Attribution-ShareAlike 4.0 International
  ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)). The raw CSV is not
  redistributed. Obtain it from the
  [original Kaggle listing](https://www.kaggle.com/datasets/ealaxi/paysim1).
- **Approved PaySim-derived scoring bundle:** distributed under CC BY-SA 4.0 with the
  PaySim attribution and modification notice in
  [Dataset and model attribution](docs/DATASET_AND_MODEL_ATTRIBUTION.md) and the
  [bundle notice](artifacts/models/paysim_fraud_model_v2_scoring/NOTICE.md).
- **Citation:** Lopez-Rojas, E. A., Elmir, A., and Axelsson, S. (2016). “PaySim: A
  Financial Mobile Money Simulator for Fraud Detection.” *28th European Modeling and
  Simulation Symposium (EMSS 2016)*, pp. 249–255.

## Downloaded model dependencies

These models are downloaded separately during local setup and are not redistributed:

- [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2):
  Apache License 2.0.
- `llama3.2:3b`: Meta Llama 3.2 Community License and
  [Acceptable Use Policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md),
  obtained through Ollama.

## Direct software dependencies

Material direct dependencies declared by the project include software under the
following license families:

- **Apache License 2.0:** Apache Airflow, Apache Kafka, Chroma, Confluent Kafka client,
  dbt-postgres, MLflow, PyArrow, Sentence Transformers, Streamlit, and XGBoost.
- **MIT License:** FastAPI, LangChain packages, LangSmith, and pytest.
- **BSD licences:** HTTPX, Jupyter/IPython components, NumPy, pandas, scikit-learn,
  SciPy, seaborn, and Uvicorn.
- **LGPL-3.0-only:** psycopg and psycopg-binary.
- **Other permissive terms:** Matplotlib uses the Matplotlib/PSF-style license;
  packaging is available under Apache 2.0 or BSD terms.

This is a concise notice for the project's direct dependencies, not a substitute for the
complete transitive dependency licenses installed by the package manager.

## Container images and services

The project references official PostgreSQL, Apache Kafka, Apache Airflow, and Python
container images. The software and base operating-system packages in those images remain
under the licenses supplied by their respective maintainers. The repository Evaluation
License does not apply to those images or services.
