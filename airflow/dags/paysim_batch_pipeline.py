"""Airflow DAG coordinating the existing PaySim batch analytics pipeline."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
from airflow.sdk import dag, task

from src.data.paysim_ingestion import (
    KNOWN_PAYSIM_TOTALS,
    IngestionSummary,
    RecordCounts,
    ingest_paysim,
)
from src.data.pipeline_validation import validate_pipeline_state
from src.data.postgres import load_postgres_settings


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))
DBT_EXECUTABLE = Path(os.getenv("DBT_EXECUTABLE", "/opt/dbt-venv/bin/dbt"))


def _summary_to_dict(summary: IngestionSummary) -> dict[str, object]:
    """Return only the small JSON-safe values needed by the next task."""
    source_totals = summary.source_totals
    database_totals = summary.database_totals
    return {
        "dataset_id": summary.dataset_id,
        "source_file_name": summary.source_file_name,
        "processed": summary.processed,
        "inserted": summary.inserted,
        "skipped": summary.skipped,
        "failed": summary.failed,
        "reconciled": summary.reconciled,
        "source_totals": source_totals.__dict__ if source_totals else None,
        "database_totals": database_totals.__dict__ if database_totals else None,
    }


def _summary_from_dict(values: dict[str, object]) -> IngestionSummary:
    """Rebuild the summary without sending millions of source rows through Airflow."""
    summary = IngestionSummary(
        dataset_id=str(values["dataset_id"]),
        source_file_name=str(values["source_file_name"]),
        processed=int(values["processed"]),
        inserted=int(values["inserted"]),
        skipped=int(values["skipped"]),
        failed=int(values["failed"]),
        reconciled=bool(values["reconciled"]),
    )
    for field_name in ("source_totals", "database_totals"):
        counts = values[field_name]
        if counts:
            setattr(summary, field_name, RecordCounts(**counts))
    return summary


@dag(
    dag_id="paysim_batch_analytics",
    description="Ingest, validate, and transform the complete PaySim dataset.",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["paysim", "batch", "analytics"],
)
def paysim_batch_analytics():
    """Define recoverable boundaries around existing ingestion and dbt behavior."""

    @task(retries=3, retry_delay=timedelta(seconds=30))
    def check_postgres_ready() -> None:
        """Retry transient startup failures before expensive ingestion begins."""
        settings = load_postgres_settings()
        with psycopg.connect(**settings.connection_kwargs(), connect_timeout=10) as connection:
            table_exists = connection.execute(
                "SELECT to_regclass('raw.transactions') IS NOT NULL"
            ).fetchone()[0]
        if not table_exists:
            raise RuntimeError("Required table raw.transactions does not exist")
        LOGGER.info("Fraud PostgreSQL database and raw transaction table are ready")

    @task(retries=1, retry_delay=timedelta(minutes=1))
    def ingest_paysim_task() -> dict[str, object]:
        """Call the Phase 3 loader so its exact validation/idempotency contract is reused."""
        summary = ingest_paysim()
        LOGGER.info("PaySim ingestion complete: %s", summary.format())
        return _summary_to_dict(summary)

    @task(retries=0)
    def validate_ingestion_task(summary_values: dict[str, object]) -> dict[str, object]:
        """Fail fast so the default all-success dependency prevents an unsafe dbt run."""
        summary = _summary_from_dict(summary_values)
        settings = load_postgres_settings()
        with psycopg.connect(**settings.connection_kwargs()) as connection:
            result = validate_pipeline_state(connection, summary, KNOWN_PAYSIM_TOTALS)
        LOGGER.info("Post-ingestion validation passed: %s", result)
        return {
            "dataset_id": result.dataset_id,
            "database_totals": result.database_totals.__dict__,
        }

    @task(retries=1, retry_delay=timedelta(minutes=1))
    def run_dbt_build(validation_result: dict[str, object]) -> None:
        """Run the existing Phase 5 project only after the independent quality gate passes."""
        LOGGER.info(
            "Starting dbt build for validated dataset %s",
            validation_result["dataset_id"],
        )
        command = [
            str(DBT_EXECUTABLE),
            "build",
            "--project-dir",
            str(PROJECT_ROOT / "dbt"),
            "--profiles-dir",
            str(PROJECT_ROOT / "dbt"),
            "--target-path",
            "/tmp/dbt-target",
            "--log-path",
            "/tmp/dbt-logs",
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            LOGGER.info("dbt output:\n%s", completed.stdout)
        if completed.returncode:
            LOGGER.error("dbt failed:\n%s", completed.stderr)
            raise RuntimeError(f"dbt build failed with exit code {completed.returncode}")
        LOGGER.info("dbt build completed for %s", json.dumps(validation_result))

    database_ready = check_postgres_ready()
    ingestion_summary = ingest_paysim_task()
    validation_result = validate_ingestion_task(ingestion_summary)
    dbt_result = run_dbt_build(validation_result)

    database_ready >> ingestion_summary
    validation_result >> dbt_result


paysim_batch_analytics()
