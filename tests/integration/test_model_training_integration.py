"""End-to-end synthetic integration test for all PH7-02 candidate models."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from mlflow import MlflowClient

from src.features.contract import REPOSITORY_ROOT
from src.models.artifact import load_model_bundle
from src.models.config import load_model_config
from src.models.train_paysim_models import train_select_and_persist


pytestmark = pytest.mark.integration


@pytest.fixture
def short_model_root():
    """Keep MLflow's nested Windows paths below the legacy length limit."""
    path = REPOSITORY_ROOT / ".pytest_tmp" / f"m-{uuid4().hex[:8]}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_contract(path: Path) -> dict:
    source = Path("config/features/paysim_v1.json")
    document = json.loads(source.read_text(encoding="utf-8"))
    counts = {"train": (8, 2), "validation": (6, 2), "test": (6, 3)}
    for split in document["splits"]:
        split["expected_rows"], split["expected_fraud_rows"] = counts[split["name"]]
    document["expected_dataset"] = {
        "total_rows": 20,
        "fraud_rows": 7,
        "final_day_rows": 1,
        "final_day_fraud_rows": 1,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return document


def _feature_frame(rows: int, fraud_indexes: set[int]) -> pd.DataFrame:
    transaction_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    types = [transaction_types[index % len(transaction_types)] for index in range(rows)]
    amounts = [float(1 + index * 2 + (50 if index in fraud_indexes else 0)) for index in range(rows)]
    return pd.DataFrame(
        {
            "amount_log1p": pd.Series(amounts).map(__import__("numpy").log1p).astype("float32"),
            **{
                f"transaction_type_{name.lower()}": pd.Series(
                    [int(value == name.upper()) for value in types], dtype="int8"
                )
                for name in ("cash_in", "cash_out", "debit", "payment", "transfer")
            },
        }
    )


def _write_partition(
    root: Path,
    name: str,
    rows: int,
    fraud_indexes: set[int],
    *,
    include_final_day: bool = False,
) -> None:
    features = _feature_frame(rows, fraud_indexes)
    target = pd.DataFrame(
        {"is_fraud": pd.Series([int(index in fraud_indexes) for index in range(rows)], dtype="int8")}
    )
    metadata = pd.DataFrame(
        {
            "transaction_id": [f"{name}-{index}" for index in range(rows)],
            "source_dataset_id": ["a" * 64] * rows,
            "source_row_number": list(range(1, rows + 1)),
            "step": list(range(1, rows + 1)),
            "simulation_day": [1] * rows,
            "partition": [name] * rows,
            "evaluation_cohort": ["standard_period"] * rows,
        }
    )
    if include_final_day:
        metadata.loc[rows - 1, ["step", "simulation_day", "evaluation_cohort"]] = [
            721,
            31,
            "final_day_artifact",
        ]
    for group, frame in (("features", features), ("target", target), ("metadata", metadata)):
        directory = root / name / group
        directory.mkdir(parents=True)
        frame.to_parquet(directory / "part-00001.parquet", index=False)


def _write_manifest(root: Path, contract: dict) -> None:
    manifest = {
        "contract_version": contract["contract_version"],
        "source_relation": contract["source_relation"],
        "source_dataset_id": "a" * 64,
        "feature_schema": [
            {"name": feature["name"], "dtype": feature["dtype"]}
            for feature in contract["features"]
        ],
        "target": contract["target_column"],
        "splits": {
            split["name"]: {
                "first_step": split["first_step"],
                "last_step": split["last_step"],
                "rows": split["expected_rows"],
                "fraud_rows": split["expected_fraud_rows"],
                "final_day_rows": 1 if split["name"] == "test" else 0,
                "final_day_fraud_rows": 1 if split["name"] == "test" else 0,
            }
            for split in contract["splits"]
        },
        "final_day_policy": {
            **contract["final_day"],
            "predictive_feature": False,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_complete_training_tracking_selection_and_persistence(tmp_path, short_model_root):
    contract_path = tmp_path / "feature_contract.json"
    contract = _write_contract(contract_path)
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _write_manifest(dataset_root, contract)
    _write_partition(dataset_root, "train", 8, {5, 7})
    _write_partition(dataset_root, "validation", 6, {4, 5})
    _write_partition(dataset_root, "test", 6, {3, 4, 5}, include_final_day=True)

    default = load_model_config()
    config = replace(
        default,
        feature_contract_path=contract_path,
        dataset_root=dataset_root,
        artifact_root=short_model_root / "model",
        xgboost={
            **default.xgboost,
            "n_estimators": 10,
            "early_stopping_rounds": 3,
            "n_jobs": 1,
        },
    )

    report = train_select_and_persist(
        config,
        tracking_root=short_model_root / "tracking",
        figures_root=short_model_root / "figures",
        validate_expected_counts=True,
    )

    assert set(report["validation_candidates"]) == {
        "no_skill",
        "logistic_regression",
        "xgboost",
    }
    assert report["selected_model"] in report["validation_candidates"]
    assert report["candidate_parameters"]["no_skill"]["strategy"] == "prior"
    assert report["candidate_parameters"]["logistic_regression"]["class_weight"] == "balanced"
    assert report["candidate_parameters"]["xgboost"]["scale_pos_weight"] == 3.0
    assert report["test_cohorts"]["day_31_artifact"]["pr_auc"] is None
    assert report["test_cohorts"]["day_31_artifact"]["roc_auc"] is None
    assert report["persistence_probability_max_absolute_error"] == 0.0
    assert (config.artifact_root / "threshold_comparison.csv").is_file()
    assert (short_model_root / "figures" / "08_test_cohort_evaluation.png").is_file()
    assert load_model_bundle(config.artifact_root).selected_model == report["selected_model"]

    tracking_database = short_model_root / "tracking" / "mlflow.db"
    client = MlflowClient(tracking_uri=f"sqlite:///{tracking_database.as_posix()}")
    experiment = client.get_experiment_by_name(config.experiment_name)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 3
    by_candidate = {run.data.tags["candidate"]: run for run in runs}
    for name in ("no_skill", "logistic_regression", "xgboost"):
        run = by_candidate[name]
        assert run.data.params["feature_contract_version"] == "paysim_features_v1"
        assert "validation_pr_auc" in run.data.metrics
        assert "validation_fraud_precision" in run.data.metrics
        assert "validation_fraud_recall" in run.data.metrics
        assert "validation_fraud_f1" in run.data.metrics
        artifact_names = {
            artifact.path for artifact in client.list_artifacts(run.info.run_id, "validation")
        }
        assert "validation/threshold_comparison.csv" in artifact_names
        assert "validation/candidate_estimator.joblib" in artifact_names
    selected_run = by_candidate[report["selected_model"]]
    assert selected_run.data.tags["selected"] == "true"
    assert "test_full_fraud_precision" in selected_run.data.metrics
    assert "test_standard_period_fraud_recall" in selected_run.data.metrics
