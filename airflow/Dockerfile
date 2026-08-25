FROM apache/airflow:3.3.1-python3.12

USER root

# dbt lives in its own environment so its dependency graph cannot alter Airflow itself.
RUN python -m venv /opt/dbt-venv \
    && /opt/dbt-venv/bin/pip install --no-cache-dir "dbt-postgres==1.11.0"

USER airflow

COPY --chown=airflow:0 airflow/requirements.txt /tmp/airflow-requirements.txt
RUN pip install --no-cache-dir -r /tmp/airflow-requirements.txt
