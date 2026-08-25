"""Airflow-specific structure tests for the PaySim batch DAG."""

from pathlib import Path

import pytest


# Standard Python CI stays lightweight; these tests run in the pinned Airflow image.
airflow_models = pytest.importorskip("airflow.models")
DagBag = airflow_models.DagBag


DAG_ID = "paysim_batch_analytics"


@pytest.fixture(scope="module")
def dag_bag() -> DagBag:
    dags_path = Path(__file__).parents[1] / "dags"
    return DagBag(dag_folder=str(dags_path))


def test_dag_imports_without_errors(dag_bag):
    assert dag_bag.import_errors == {}
    assert DAG_ID in dag_bag.dags


def test_dag_has_expected_manual_run_policy(dag_bag):
    pipeline = dag_bag.dags[DAG_ID]

    assert pipeline.schedule is None
    assert pipeline.catchup is False
    assert pipeline.max_active_runs == 1


def test_dag_uses_recoverable_task_boundaries_in_order(dag_bag):
    pipeline = dag_bag.dags[DAG_ID]
    expected_ids = {
        "check_postgres_ready",
        "ingest_paysim_task",
        "validate_ingestion_task",
        "run_dbt_build",
    }

    assert set(pipeline.task_ids) == expected_ids
    assert pipeline.get_task("check_postgres_ready").downstream_task_ids == {
        "ingest_paysim_task"
    }
    assert pipeline.get_task("ingest_paysim_task").downstream_task_ids == {
        "validate_ingestion_task"
    }
    assert pipeline.get_task("validate_ingestion_task").downstream_task_ids == {
        "run_dbt_build"
    }


def test_retry_and_failure_blocking_policies_are_explicit(dag_bag):
    pipeline = dag_bag.dags[DAG_ID]

    assert pipeline.get_task("check_postgres_ready").retries == 3
    assert pipeline.get_task("ingest_paysim_task").retries == 1
    assert pipeline.get_task("validate_ingestion_task").retries == 0
    assert pipeline.get_task("run_dbt_build").trigger_rule.value == "all_success"
