"""Load the small, reviewed evidence summary shown beside live model scores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.api.config import load_api_settings


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_INFORMATION_PATH = (
    REPOSITORY_ROOT / "config" / "dashboard" / "fraud_model_information.json"
)


class ModelInformationError(ValueError):
    """Raised when the reviewed model-information snapshot is inconsistent."""


@dataclass(frozen=True)
class ModelEvaluationInformation:
    """Verified validation evidence for the deployed v2 estimator."""

    label: str
    population: str
    role: str
    rows: int
    fraud_rows: int
    legitimate_rows: int
    fraud_rate: float
    threshold: float
    accuracy: float
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


@dataclass(frozen=True)
class FraudModelInformation:
    """Identity, threshold, and explicitly labelled validation evidence."""

    model_version: str
    model_type: str
    selected_run_id: str
    feature_contract_version: str
    validation_evaluation: ModelEvaluationInformation
    held_out_test_evaluation: ModelEvaluationInformation
    decision_threshold: float
    threshold_status: str
    v2_test_evaluation_available: bool
    validation_evidence_source: str
    held_out_test_evidence_source: str


def _require_probability(name: str, value: Any) -> float:
    numeric = float(value)
    if not 0 <= numeric <= 1:
        raise ModelInformationError(f"{name} must be between zero and one")
    return numeric


def load_model_information(
    path: str | Path = DEFAULT_MODEL_INFORMATION_PATH,
) -> FraudModelInformation:
    """Load reviewed metadata and reject count or metric drift before display."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        def evaluation_information(name: str) -> ModelEvaluationInformation:
            evaluation = document[name]
            return ModelEvaluationInformation(
                label=str(evaluation["label"]),
                population=str(evaluation["population"]),
                role=str(evaluation["role"]),
                rows=int(evaluation["rows"]),
                fraud_rows=int(evaluation["fraud_rows"]),
                legitimate_rows=int(evaluation["legitimate_rows"]),
                fraud_rate=_require_probability("fraud_rate", evaluation["fraud_rate"]),
                threshold=_require_probability("threshold", evaluation["threshold"]),
                accuracy=_require_probability("accuracy", evaluation["accuracy"]),
                pr_auc=_require_probability("pr_auc", evaluation["pr_auc"]),
                roc_auc=_require_probability("roc_auc", evaluation["roc_auc"]),
                precision=_require_probability("precision", evaluation["precision"]),
                recall=_require_probability("recall", evaluation["recall"]),
                f1=_require_probability("f1", evaluation["f1"]),
                true_negatives=int(evaluation["true_negatives"]),
                false_positives=int(evaluation["false_positives"]),
                false_negatives=int(evaluation["false_negatives"]),
                true_positives=int(evaluation["true_positives"]),
            )

        information = FraudModelInformation(
            model_version=str(document["model_version"]),
            model_type=str(document["model_type"]),
            selected_run_id=str(document["selected_run_id"]),
            feature_contract_version=str(document["feature_contract_version"]),
            validation_evaluation=evaluation_information("validation_evaluation"),
            held_out_test_evaluation=evaluation_information(
                "held_out_test_evaluation"
            ),
            decision_threshold=_require_probability(
                "decision_threshold", document["decision_threshold"]
            ),
            threshold_status=str(document["threshold_status"]),
            v2_test_evaluation_available=bool(
                document["v2_test_evaluation_available"]
            ),
            validation_evidence_source=str(document["validation_evidence_source"]),
            held_out_test_evidence_source=str(
                document["held_out_test_evidence_source"]
            ),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, ModelInformationError):
            raise
        raise ModelInformationError("Model information is unavailable or invalid") from error

    for observed in (
        information.validation_evaluation,
        information.held_out_test_evaluation,
    ):
        if observed.fraud_rows + observed.legitimate_rows != observed.rows:
            raise ModelInformationError("Evaluation class counts do not match row count")
        if (
            observed.true_negatives
            + observed.false_positives
            + observed.false_negatives
            + observed.true_positives
            != observed.rows
        ):
            raise ModelInformationError("Confusion-matrix counts do not match row count")
        if not np.isclose(
            observed.fraud_rate,
            observed.fraud_rows / observed.rows,
            rtol=0,
            atol=1e-12,
        ):
            raise ModelInformationError("Evaluation prevalence does not match class counts")
    if not information.v2_test_evaluation_available:
        raise ModelInformationError("The final v2 held-out evaluation must be present")
    api = load_api_settings()
    if (
        information.model_version != api.model_version
        or information.selected_run_id != api.selected_run_id
        or information.feature_contract_version != api.feature_contract_version
        or information.decision_threshold != api.risk_policy.high_min_probability
        or information.validation_evaluation.threshold != information.decision_threshold
        or information.held_out_test_evaluation.threshold
        != information.decision_threshold
    ):
        raise ModelInformationError(
            "Model information does not match the deployed API configuration"
        )
    return information
