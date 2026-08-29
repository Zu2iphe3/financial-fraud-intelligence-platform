"""Record reproducible candidate evidence in the planned local MLflow store."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import joblib
import mlflow
import pandas as pd
from mlflow import MlflowClient

from src.features.contract import REPOSITORY_ROOT
from src.models.candidates import TrainedCandidate, model_explanation
from src.models.evaluation import ThresholdSelection


DEFAULT_TRACKING_ROOT = REPOSITORY_ROOT / "mlruns"


def configure_tracking(experiment_name: str, tracking_root: str | Path = DEFAULT_TRACKING_ROOT) -> None:
    """Use ignored local SQLite tracking state so no external account is required."""
    root = Path(tracking_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    database_path = root / "mlflow.db"
    # Current MLflow releases place the legacy filesystem tracking backend in
    # maintenance mode. SQLite remains local while supporting current MLflow behavior.
    mlflow.set_tracking_uri(f"sqlite:///{database_path.as_posix()}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=(root / "artifacts").as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)


def _numeric_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Exclude undefined one-class values while retaining every numeric decision metric."""
    logged: dict[str, float] = {}
    for name, value in metrics.items():
        if value is not None and isinstance(value, (int, float)):
            logged[f"{prefix}{name}"] = float(value)
    return logged


def log_candidate_run(
    candidate: TrainedCandidate,
    validation_metrics: Mapping[str, Any],
    threshold_selection: ThresholdSelection,
    feature_names: tuple[str, ...],
    common_parameters: Mapping[str, Any],
    *,
    run_name: str | None = None,
    extra_tags: Mapping[str, str] | None = None,
    extra_metrics: Mapping[str, Any] | None = None,
    extra_tables: Mapping[str, pd.DataFrame] | None = None,
) -> str:
    """Store one comparable validation run and return its immutable MLflow run ID."""
    with mlflow.start_run(run_name=run_name or candidate.name) as run:
        tags = {
            "candidate": candidate.name,
            "selection_partition": "validation",
            "primary_selection_metric": "validation_pr_auc",
            "accuracy_role": "informational_only",
            "roc_auc_role": "secondary",
        }
        tags.update(dict(extra_tags or {}))
        mlflow.set_tags(tags)
        mlflow.log_params({**dict(common_parameters), **candidate.tracked_parameters})
        mlflow.log_metrics(_numeric_metrics(validation_metrics, "validation_"))
        mlflow.log_metrics(_numeric_metrics(extra_metrics or {}))
        with tempfile.TemporaryDirectory(prefix="paysim-mlflow-") as temporary:
            root = Path(temporary)
            threshold_selection.comparison.to_csv(root / "threshold_comparison.csv", index=False)
            threshold_selection.curve.to_csv(root / "threshold_curve.csv", index=False)
            explanation = model_explanation(candidate, feature_names)
            explanation.to_csv(root / "model_explanation.csv", index=False)
            (root / "validation_metrics.json").write_text(
                json.dumps(dict(validation_metrics), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            joblib.dump(candidate.estimator, root / "candidate_estimator.joblib")
            for filename, table in (extra_tables or {}).items():
                table.to_csv(root / filename, index=False)
            mlflow.log_artifacts(str(root), artifact_path="validation")
        return run.info.run_id


def log_selected_test_results(
    run_id: str,
    cohort_metrics: Mapping[str, Mapping[str, Any]],
    selected_threshold: float,
    artifact_directory: str | Path,
) -> None:
    """Resume only the locked winner run after final test evaluation."""
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags(
            {
                "selected": "true",
                "selection_locked_before_test": "true",
                "day_31_policy": "separate_one_class_artifact_cohort",
            }
        )
        mlflow.log_param("selected_threshold", selected_threshold)
        for cohort, metrics in cohort_metrics.items():
            mlflow.log_metrics(_numeric_metrics(metrics, f"test_{cohort}_"))
        mlflow.log_artifacts(str(artifact_directory), artifact_path="selected_model")
