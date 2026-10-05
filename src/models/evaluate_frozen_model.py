"""Evaluate the frozen behavioural-v2 scoring artifact once on chronological test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.api.config import ApiSettings, load_api_settings
from src.features.contract import load_feature_contract
from src.models.data import (
    load_dataset_manifest,
    load_partition,
    validate_dataset_manifest,
)
from src.models.evaluation import combined_evaluation
from src.models.scoring_artifact import ScoringModelBundle, load_scoring_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPOSITORY_ROOT / "data/processed/paysim_features/paysim_features_v2"
SEARCH_EVIDENCE_PATH = (
    REPOSITORY_ROOT / "artifacts/model_reviews/paysim_xgboost_search_v2/search_review.json"
)
VISUAL_EVIDENCE_PATH = (
    REPOSITORY_ROOT / "artifacts/model_reviews/paysim_v2_final_visuals/visual_manifest.json"
)
FEATURE_EVIDENCE_PATHS = (
    REPOSITORY_ROOT
    / "artifacts/model_reviews/paysim_v2_validation_comparison/comparison.json",
    REPOSITORY_ROOT / "artifacts/model_reviews/paysim_v2_final_baselines/comparison.json",
)
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "artifacts/model_reviews/paysim_v2_frozen_test_evaluation/evaluation.json"
)


class FrozenModelEvaluationError(RuntimeError):
    """Raised when frozen identity or held-out test guarantees cannot be proved."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenModelEvaluationError(f"Cannot load evaluation evidence {path}") from error
    if not isinstance(document, dict):
        raise FrozenModelEvaluationError(f"Evaluation evidence {path} must be an object")
    return document


def _verify_frozen_identity(
    settings: ApiSettings,
    bundle: ScoringModelBundle,
    search: dict[str, Any],
    visuals: dict[str, Any],
    feature_reviews: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Prove model and threshold selection finished before test data is opened."""
    winner_name = search.get("winner")
    winner = search.get("trials", {}).get(winner_name, {})
    validation = winner.get("validation_metrics", {})
    expected_identity = (
        settings.model_version,
        settings.feature_contract_version,
        settings.feature_contract_sha256,
        settings.feature_names,
        settings.selected_model,
        settings.selected_run_id,
    )
    if bundle.manifest_identity != expected_identity:
        raise FrozenModelEvaluationError("Persisted scoring artifact does not match API identity")
    if (
        search.get("selection_partition") != "validation"
        or search.get("test_partition_loaded") is not False
        or winner_name != "high_child_weight"
        or winner.get("run_id") != settings.selected_run_id
        or not np.isclose(
            float(validation.get("threshold", -1)),
            settings.risk_policy.high_min_probability,
            rtol=0,
            atol=1e-12,
        )
    ):
        raise FrozenModelEvaluationError("Search evidence does not prove a frozen validation choice")
    if (
        visuals.get("selection_partition") != "validation"
        or visuals.get("test_partition_loaded") is not False
        or visuals.get("tuned_run_id") != settings.selected_run_id
    ):
        raise FrozenModelEvaluationError("Visual evidence does not preserve test isolation")
    if len(feature_reviews) != len(FEATURE_EVIDENCE_PATHS) or any(
        review.get("selection_partition") != "validation"
        or review.get("test_partition_loaded") is not False
        for review in feature_reviews
    ):
        raise FrozenModelEvaluationError("Feature-review evidence does not preserve test isolation")
    return {
        "model_fitting_used_test": False,
        "hyperparameter_selection_used_test": False,
        "feature_selection_used_test": False,
        "threshold_selection_used_test": False,
        "selection_partition": "validation",
        "selection_run_id": settings.selected_run_id,
        "selection_evidence": str(SEARCH_EVIDENCE_PATH.relative_to(REPOSITORY_ROOT)),
        "visual_evidence": str(VISUAL_EVIDENCE_PATH.relative_to(REPOSITORY_ROOT)),
        "feature_review_evidence": [
            str(path.relative_to(REPOSITORY_ROOT)) for path in FEATURE_EVIDENCE_PATHS
        ],
    }


def _evaluate_cohorts(
    target: np.ndarray,
    probabilities: np.ndarray,
    evaluation_cohorts: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Report the predefined full, ordinary-period, and Day 31 test cohorts."""
    masks = {
        "full_test": np.ones(len(target), dtype=bool),
        "standard_period_days_25_30": evaluation_cohorts == "standard_period",
        "day_31_artifact": evaluation_cohorts == "final_day_artifact",
    }
    results: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        if not mask.any():
            raise FrozenModelEvaluationError(f"Held-out cohort {name} is empty")
        results[name] = combined_evaluation(target[mask], probabilities[mask], threshold)
    results["day_31_artifact"]["ranking_metrics_informative"] = False
    results["day_31_artifact"]["limitation"] = (
        "All Day 31 rows are fraud, so PR-AUC and ROC-AUC are not informative."
    )
    return results


def evaluate_frozen_model(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Evaluate without fitting, tuning, feature selection, or threshold selection."""
    destination = Path(output_path)
    if destination.exists():
        raise FrozenModelEvaluationError(
            f"Final held-out report already exists: {destination}"
        )

    settings = load_api_settings()
    contract = load_feature_contract("config/features/paysim_v2.json")
    manifest = load_dataset_manifest(DATASET_ROOT)
    validate_dataset_manifest(manifest, contract)
    bundle = load_scoring_bundle(settings.model_artifact_path)
    isolation = _verify_frozen_identity(
        settings,
        bundle,
        _load_json(SEARCH_EVIDENCE_PATH),
        _load_json(VISUAL_EVIDENCE_PATH),
        tuple(_load_json(path) for path in FEATURE_EVIDENCE_PATHS),
    )

    # The held-out partition is intentionally opened only after every selection and
    # deployment identity above has been locked and verified.
    test = load_partition(DATASET_ROOT, "test", contract)
    if (
        int(test.metadata["step"].min()) != 577
        or int(test.metadata["step"].max()) != 743
        or not test.metadata["simulation_day"].between(25, 31).all()
    ):
        raise FrozenModelEvaluationError("Held-out population is not steps 577-743")

    probabilities = bundle.predict_probabilities(test.features)
    threshold = settings.risk_policy.high_min_probability
    report = {
        "evaluation_status": "final_one_time_held_out_evaluation",
        "model_version": settings.model_version,
        "model_type": "XGBoost classifier",
        "selected_model": settings.selected_model,
        "selected_run_id": settings.selected_run_id,
        "feature_contract_version": settings.feature_contract_version,
        "feature_contract_sha256": settings.feature_contract_sha256,
        "frozen_threshold": threshold,
        "threshold_source": "validation maximum-F1 review point",
        "threshold_changed_after_test": False,
        "test_population": {
            "partition": "test",
            "first_step": 577,
            "last_step": 743,
            "first_day": 25,
            "last_day": 31,
            "source_dataset_id": manifest["source_dataset_id"],
        },
        "test_isolation_verification": isolation,
        "metric_semantics": {
            "threshold_independent": ["pr_auc", "roc_auc"],
            "frozen_threshold": [
                "accuracy",
                "fraud_precision",
                "fraud_recall",
                "fraud_f1",
                "true_negatives",
                "false_positives",
                "false_negatives",
                "true_positives",
                "alerts",
            ],
        },
        "cohorts": _evaluate_cohorts(
            test.target.to_numpy(dtype=np.int8),
            probabilities,
            test.metadata["evaluation_cohort"].astype(str).to_numpy(),
            threshold,
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    try:
        report = evaluate_frozen_model(arguments.output)
    except (OSError, ValueError, FrozenModelEvaluationError) as error:
        print(f"Held-out evaluation failed: {error}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
