"""Promote the reviewed PH7-02 MLflow estimator without training or test evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from mlflow import MlflowClient

from src.features.contract import load_feature_contract
from src.models.data import contract_sha256
from src.models.scoring_artifact import ScoringModelBundle, save_scoring_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_CONFIG = REPOSITORY_ROOT / "config/api/fraud_prediction.json"
TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")


class ModelPromotionError(ValueError):
    """Raised when MLflow evidence does not identify the approved v2 estimator."""


def _repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _load_model_settings(path: str | Path) -> dict[str, str]:
    """Read only model-promotion fields from the shared API configuration."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        model = document["model"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ModelPromotionError("API model configuration is missing or invalid") from error
    required = {
        "artifact_path",
        "feature_contract_path",
        "model_version",
        "selected_model",
        "mlflow_run_id",
        "mlflow_artifact_path",
        "mlflow_feature_names_artifact_path",
        "mlflow_feature_contract_sha256",
    }
    if not required.issubset(model) or not all(str(model[key]).strip() for key in required):
        raise ModelPromotionError("API model configuration is incomplete")
    return {key: str(model[key]) for key in required}


def _validate_run(run: object, settings: dict[str, str], run_contract_hash: str) -> None:
    """Prevent a different, test-touched, or stale experiment from being promoted."""
    params = run.data.params
    tags = run.data.tags
    expected = {
        "feature_contract_version": "paysim_features_v2",
        "feature_contract_sha256": run_contract_hash,
        "test_partition_loaded": "False",
    }
    mismatches = [key for key, value in expected.items() if params.get(key) != value]
    if tags.get("trial_name") != "high_child_weight":
        mismatches.append("trial_name")
    if run.info.run_id != settings["mlflow_run_id"]:
        mismatches.append("mlflow_run_id")
    if mismatches:
        raise ModelPromotionError(
            "MLflow run does not match the approved PH7-02 evidence: " + ", ".join(mismatches)
        )


def promote_selected_model(
    *,
    config_path: str | Path = DEFAULT_API_CONFIG,
    tracking_root: str | Path = REPOSITORY_ROOT / "mlruns",
    output_path: str | Path | None = None,
) -> Path:
    """Package the fitted estimator from MLflow without reading modelling datasets."""
    settings = _load_model_settings(config_path)
    contract_path = _repository_path(settings["feature_contract_path"])
    contract = load_feature_contract(contract_path)
    contract_hash = contract_sha256(contract_path)
    database = Path(tracking_root).resolve() / "mlflow.db"
    if not database.is_file():
        raise ModelPromotionError("Local MLflow database is unavailable")
    client = MlflowClient(f"sqlite:///{database.as_posix()}")
    try:
        run = client.get_run(settings["mlflow_run_id"])
    except Exception as error:  # MLflow backends expose store-specific exceptions.
        raise ModelPromotionError("Approved MLflow run is unavailable") from error
    _validate_run(run, settings, settings["mlflow_feature_contract_sha256"])

    with tempfile.TemporaryDirectory(prefix="paysim-model-promotion-") as temporary:
        try:
            estimator_path = client.download_artifacts(
                run.info.run_id, settings["mlflow_artifact_path"], temporary
            )
            explanation_path = client.download_artifacts(
                run.info.run_id,
                settings["mlflow_feature_names_artifact_path"],
                temporary,
            )
            estimator = joblib.load(estimator_path)
            recorded_features = set(pd.read_csv(explanation_path)["feature"].astype(str))
        except Exception as error:
            raise ModelPromotionError("Approved estimator artifact cannot be loaded") from error

    if int(getattr(estimator, "n_features_in_", -1)) != len(contract.feature_names):
        raise ModelPromotionError("Estimator feature count does not match the v2 contract")
    if recorded_features != set(contract.feature_names):
        raise ModelPromotionError("MLflow feature evidence does not match the v2 contract")
    package_versions = {
        package: importlib.metadata.version(package) for package in TRACKED_PACKAGES
    }
    bundle = ScoringModelBundle(
        model_version=settings["model_version"],
        feature_contract_version=contract.contract_version,
        feature_contract_sha256=contract_hash,
        feature_names=contract.feature_names,
        estimator=estimator,
        selected_model=settings["selected_model"],
        selected_run_id=run.info.run_id,
        package_versions=package_versions,
    )
    destination = Path(output_path) if output_path else _repository_path(settings["artifact_path"])
    return save_scoring_bundle(bundle, destination)


def main() -> int:
    """Expose promotion as a reproducible command with explicit local paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_API_CONFIG))
    parser.add_argument("--tracking-root", default=str(REPOSITORY_ROOT / "mlruns"))
    parser.add_argument("--output")
    arguments = parser.parse_args()
    try:
        path = promote_selected_model(
            config_path=arguments.config,
            tracking_root=arguments.tracking_root,
            output_path=arguments.output,
        )
    except (OSError, ValueError) as error:
        print(f"Model promotion failed: {error}")
        return 1
    print(f"Promoted reviewed scoring model to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
