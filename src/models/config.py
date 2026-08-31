"""Load and validate the reproducible PH7-02 modelling configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.features.contract import REPOSITORY_ROOT


DEFAULT_MODEL_CONFIG_PATH = REPOSITORY_ROOT / "config" / "models" / "paysim_fraud_v1.json"


class ModelConfigError(ValueError):
    """Raised when model selection cannot be reproduced from the configuration."""


@dataclass(frozen=True)
class ModelConfig:
    """Settings that define candidate training, selection, and persisted outputs."""

    model_version: str
    experiment_name: str
    feature_contract_path: Path
    dataset_root: Path
    artifact_root: Path
    random_seed: int
    primary_selection_metric: str
    threshold_objective: str
    threshold_tie_breakers: tuple[str, ...]
    reference_thresholds: tuple[float, ...]
    reference_recall_targets: tuple[float, ...]
    logistic_regression: Mapping[str, Any]
    xgboost: Mapping[str, Any]
    probability_parity_tolerance: float


def _repository_path(value: str) -> Path:
    """Resolve committed relative paths without embedding a developer machine path."""
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_model_config(path: str | Path = DEFAULT_MODEL_CONFIG_PATH) -> ModelConfig:
    """Load the committed model configuration and reject unsafe selection changes."""
    config_path = _repository_path(str(path))
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
        threshold = document["threshold_policy"]
        config = ModelConfig(
            model_version=str(document["model_version"]),
            experiment_name=str(document["experiment_name"]),
            feature_contract_path=_repository_path(document["feature_contract_path"]),
            dataset_root=_repository_path(document["dataset_root"]),
            artifact_root=_repository_path(document["artifact_root"]),
            random_seed=int(document["random_seed"]),
            primary_selection_metric=str(document["primary_selection_metric"]),
            threshold_objective=str(threshold["objective"]),
            threshold_tie_breakers=tuple(threshold["tie_breakers"]),
            reference_thresholds=tuple(float(value) for value in threshold["reference_thresholds"]),
            reference_recall_targets=tuple(
                float(value) for value in threshold["reference_recall_targets"]
            ),
            logistic_regression=dict(document["logistic_regression"]),
            xgboost=dict(document["xgboost"]),
            probability_parity_tolerance=float(document["probability_parity_tolerance"]),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ModelConfigError(f"Cannot load model configuration {config_path}: {error}") from error

    problems: list[str] = []
    if config.primary_selection_metric != "validation_pr_auc":
        problems.append("validation PR-AUC must remain the primary selection metric")
    if config.threshold_objective != "fraud_f1":
        problems.append("the threshold objective must remain fraud F1")
    if config.threshold_tie_breakers != ("fraud_recall", "false_positives", "threshold"):
        problems.append("threshold tie-breakers must be recall, false positives, then threshold")
    if not config.reference_thresholds or any(
        value <= 0 or value >= 1 for value in config.reference_thresholds
    ):
        problems.append("reference thresholds must be strictly between zero and one")
    if not config.reference_recall_targets or any(
        value <= 0 or value > 1 for value in config.reference_recall_targets
    ):
        problems.append("reference recall targets must be greater than zero and at most one")
    if config.probability_parity_tolerance <= 0:
        problems.append("probability parity tolerance must be positive")
    if problems:
        raise ModelConfigError("Invalid model configuration: " + "; ".join(problems))
    return config
