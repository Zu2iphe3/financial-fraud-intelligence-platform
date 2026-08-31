"""Train, compare, track, select, and persist the PH7-02 PaySim fraud model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.contract import FeatureContract, REPOSITORY_ROOT, load_feature_contract
from src.models.artifact import FraudModelBundle, load_model_bundle, save_model_bundle
from src.models.candidates import TrainedCandidate, fraud_probabilities, train_candidate
from src.models.config import DEFAULT_MODEL_CONFIG_PATH, ModelConfig, load_model_config
from src.models.data import (
    contract_sha256,
    load_dataset_manifest,
    load_partition,
    PartitionData,
    training_class_ratio,
    validate_dataset_manifest,
)
from src.models.evaluation import (
    ThresholdSelection,
    combined_evaluation,
    evaluate_probabilities,
    select_candidate,
    select_validation_threshold,
)
from src.models.reporting import (
    plot_test_confusion_matrices,
    plot_threshold_tradeoff,
    plot_validation_precision_recall,
)
from src.models.tracking import DEFAULT_TRACKING_ROOT, configure_tracking, log_candidate_run, log_selected_test_results


LOGGER = logging.getLogger(__name__)
DEFAULT_FIGURES_ROOT = REPOSITORY_ROOT / "docs" / "figures"
CANDIDATE_NAMES = ("no_skill", "logistic_regression", "xgboost")


class ModelTrainingError(RuntimeError):
    """Raised when a complete and leakage-safe selection run cannot be produced."""


@dataclass(frozen=True)
class CandidateEvidence:
    """One fitted candidate and all validation-only evidence used for selection."""

    candidate: TrainedCandidate
    probabilities: np.ndarray
    threshold: ThresholdSelection
    metrics: dict[str, Any]
    run_id: str


def _file_sha256(path: Path) -> str:
    """Fingerprint the committed model settings attached to an experiment."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions() -> dict[str, str]:
    """Capture the runtime needed to interpret or reproduce the selected artifact."""
    packages = ("numpy", "pandas", "scikit-learn", "xgboost", "mlflow", "joblib", "pyarrow")
    observed = {"python": platform.python_version()}
    for package in packages:
        try:
            observed[package] = version(package)
        except PackageNotFoundError:
            observed[package] = "missing"
    return observed


def _common_tracking_parameters(
    config: ModelConfig,
    manifest: dict[str, Any],
    contract_digest: str,
    class_ratio: float,
) -> dict[str, Any]:
    """Attach the same dataset and selection identity to every candidate run."""
    return {
        "model_version": config.model_version,
        "feature_contract_version": manifest["contract_version"],
        "feature_contract_sha256": contract_digest,
        "source_dataset_id": manifest["source_dataset_id"],
        "random_seed": config.random_seed,
        "training_scale_pos_weight": class_ratio,
        "primary_selection_metric": config.primary_selection_metric,
        "threshold_objective": config.threshold_objective,
        "train_rows": manifest["splits"]["train"]["rows"],
        "validation_rows": manifest["splits"]["validation"]["rows"],
        "test_rows": manifest["splits"]["test"]["rows"],
    }


def _candidate_validation(
    candidate: TrainedCandidate,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    config: ModelConfig,
) -> tuple[np.ndarray, ThresholdSelection, dict[str, Any]]:
    """Calculate ranking and mandatory operating metrics on validation only."""
    probabilities = fraud_probabilities(candidate, validation_features)
    threshold = select_validation_threshold(
        validation_target,
        probabilities,
        config.reference_thresholds,
        config.reference_recall_targets,
    )
    metrics = {
        **evaluate_probabilities(validation_target, probabilities),
        **threshold.selected_metrics,
    }
    return probabilities, threshold, metrics


def _test_cohort_metrics(
    target: pd.Series,
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    artifact_cohort: str,
) -> dict[str, dict[str, Any]]:
    """Separate ordinary future behaviour from PaySim's all-fraud final day."""
    masks = {
        "full": np.ones(len(target), dtype=bool),
        "standard_period": metadata["evaluation_cohort"].eq("standard_period").to_numpy(),
        "day_31_artifact": metadata["evaluation_cohort"].eq(artifact_cohort).to_numpy(),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        if not mask.any():
            raise ModelTrainingError(f"Test cohort {name!r} is empty")
        results[name] = combined_evaluation(target.to_numpy()[mask], probabilities[mask], threshold)
        if name == "day_31_artifact":
            results[name]["ranking_metrics_informative"] = False
            results[name]["limitation"] = "All rows are fraud; PR-AUC and ROC-AUC are non-informative."
    return results


def _write_run_artifacts(
    destination: Path,
    report: dict[str, Any],
    validation_results: dict[str, dict[str, Any]],
    selected_threshold: ThresholdSelection,
) -> None:
    """Keep machine-readable decision evidence beside the selected model bundle."""
    (destination / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected_threshold.comparison.to_csv(destination / "threshold_comparison.csv", index=False)
    selected_threshold.curve.to_csv(destination / "threshold_curve.csv", index=False)
    comparison_rows = []
    for name, result in validation_results.items():
        comparison_rows.append({"candidate": name, **result})
    pd.DataFrame(comparison_rows).to_csv(destination / "validation_comparison.csv", index=False)


def _copy_documentation_figures(generated: dict[str, Path], figures_root: Path) -> None:
    """Publish only three small review figures, not local MLflow or model binaries."""
    figures_root.mkdir(parents=True, exist_ok=True)
    targets = {
        "validation_pr": figures_root / "06_model_validation_pr_curve.png",
        "threshold": figures_root / "07_selected_threshold_tradeoff.png",
        "test_cohorts": figures_root / "08_test_cohort_evaluation.png",
    }
    for name, target in targets.items():
        shutil.copyfile(generated[name], target)


def _train_and_track_candidates(
    train: PartitionData,
    validation: PartitionData,
    contract: FeatureContract,
    config: ModelConfig,
    class_ratio: float,
    common_parameters: dict[str, Any],
) -> dict[str, CandidateEvidence]:
    """Fit every candidate against identical training and validation partitions."""
    evidence: dict[str, CandidateEvidence] = {}
    for name in CANDIDATE_NAMES:
        LOGGER.info("Training candidate: %s", name)
        candidate = train_candidate(
            name,
            train.features,
            train.target,
            validation.features,
            validation.target,
            config,
            class_ratio,
        )
        probabilities, threshold, metrics = _candidate_validation(
            candidate, validation.features, validation.target, config
        )
        run_id = log_candidate_run(
            candidate, metrics, threshold, contract.feature_names, common_parameters
        )
        evidence[name] = CandidateEvidence(
            candidate=candidate,
            probabilities=probabilities,
            threshold=threshold,
            metrics=metrics,
            run_id=run_id,
        )
    return evidence


def _persistence_parity_error(
    bundle: FraudModelBundle,
    contract: FeatureContract,
    artifact_root: Path,
    tolerance: float,
) -> float:
    """Prove that saving and loading does not change fixed-example probabilities."""
    fixed_examples = pd.DataFrame(
        {
            "amount": [0.0, 100.0, 10_000.0, 1_000_000.0, 250.0],
            "transaction_type": list(contract.transaction_types),
        }
    )
    before = bundle.predict_probabilities(fixed_examples)
    after = load_model_bundle(artifact_root).predict_probabilities(fixed_examples)
    parity_error = float(np.max(np.abs(before - after)))
    if parity_error > tolerance:
        raise ModelTrainingError(
            f"Persisted probability difference {parity_error} exceeds {tolerance}"
        )
    return parity_error


def _build_evaluation_report(
    config: ModelConfig,
    contract: FeatureContract,
    manifest: dict[str, Any],
    contract_digest: str,
    class_ratio: float,
    evidence: dict[str, CandidateEvidence],
    selected_name: str,
    validation: PartitionData,
    cohort_metrics: dict[str, dict[str, Any]],
    parity_error: float,
    versions: dict[str, str],
) -> dict[str, Any]:
    """Assemble one complete, machine-readable record of the selection decision."""
    selected = evidence[selected_name]
    return {
        "model_version": config.model_version,
        "feature_contract_version": contract.contract_version,
        "feature_contract_sha256": contract_digest,
        "source_dataset_id": manifest["source_dataset_id"],
        "primary_selection_metric": "validation_pr_auc",
        "selection_interpretation": (
            "PR-AUC ranks candidates, but recall, precision, F1, false negatives, "
            "false positives, and confusion matrices are mandatory decision evidence."
        ),
        "selected_model": selected_name,
        "selected_run_id": selected.run_id,
        "selected_threshold": selected.threshold.threshold,
        "threshold_policy": (
            "maximum validation fraud F1; tie by recall, false positives, threshold"
        ),
        "threshold_status": "validation recommendation; operational approval remains required",
        "training_scale_pos_weight": class_ratio,
        "model_configuration": {
            "random_seed": config.random_seed,
            "reference_thresholds": list(config.reference_thresholds),
            "reference_recall_targets": list(config.reference_recall_targets),
            "logistic_regression": dict(config.logistic_regression),
            "xgboost": dict(config.xgboost),
        },
        "candidate_parameters": {
            name: item.candidate.tracked_parameters for name, item in evidence.items()
        },
        "validation_candidates": {
            name: item.metrics for name, item in evidence.items()
        },
        "no_skill_default_threshold": combined_evaluation(
            validation.target, evidence["no_skill"].probabilities, 0.5
        ),
        "candidate_run_ids": {name: item.run_id for name, item in evidence.items()},
        "test_cohorts": cohort_metrics,
        "persistence_probability_max_absolute_error": parity_error,
        "package_versions": versions,
    }


def _create_evaluation_figures(
    artifact_root: Path,
    validation: PartitionData,
    evidence: dict[str, CandidateEvidence],
    selected_name: str,
    cohort_metrics: dict[str, dict[str, Any]],
    figures_root: str | Path | None,
) -> None:
    """Render the three intended portfolio figures from locked evaluation evidence."""
    selected = evidence[selected_name]
    generated = {
        "validation_pr": artifact_root / "validation_pr_curve.png",
        "threshold": artifact_root / "selected_threshold_tradeoff.png",
        "test_cohorts": artifact_root / "test_cohort_confusion_matrices.png",
    }
    plot_validation_precision_recall(
        validation.target,
        {name: item.probabilities for name, item in evidence.items()},
        generated["validation_pr"],
    )
    plot_threshold_tradeoff(
        selected.threshold.curve,
        selected.threshold.threshold,
        generated["threshold"],
    )
    plot_test_confusion_matrices(cohort_metrics, generated["test_cohorts"])
    if figures_root is not None:
        _copy_documentation_figures(generated, Path(figures_root))


def _finalize_selected_model(
    config: ModelConfig,
    contract: FeatureContract,
    manifest: dict[str, Any],
    contract_digest: str,
    class_ratio: float,
    evidence: dict[str, CandidateEvidence],
    selected_name: str,
    validation: PartitionData,
    test: PartitionData,
    figures_root: str | Path | None,
) -> dict[str, Any]:
    """Evaluate the locked winner once, persist it, and publish review evidence."""
    selected = evidence[selected_name]
    test_probabilities = fraud_probabilities(selected.candidate, test.features)
    cohort_metrics = _test_cohort_metrics(
        test.target,
        test.metadata,
        test_probabilities,
        selected.threshold.threshold,
        contract.final_day.evaluation_cohort,
    )
    versions = package_versions()
    bundle = FraudModelBundle(
        model_version=config.model_version,
        feature_contract=contract,
        feature_contract_sha256=contract_digest,
        estimator=selected.candidate.estimator,
        threshold=selected.threshold.threshold,
        selected_model=selected_name,
        selected_run_id=selected.run_id,
        package_versions=versions,
    )
    save_model_bundle(bundle, config.artifact_root)
    parity_error = _persistence_parity_error(
        bundle,
        contract,
        config.artifact_root,
        config.probability_parity_tolerance,
    )
    report = _build_evaluation_report(
        config,
        contract,
        manifest,
        contract_digest,
        class_ratio,
        evidence,
        selected_name,
        validation,
        cohort_metrics,
        parity_error,
        versions,
    )
    validation_results = report["validation_candidates"]
    _write_run_artifacts(
        config.artifact_root, report, validation_results, selected.threshold
    )
    _create_evaluation_figures(
        config.artifact_root,
        validation,
        evidence,
        selected_name,
        cohort_metrics,
        figures_root,
    )
    log_selected_test_results(
        selected.run_id,
        cohort_metrics,
        selected.threshold.threshold,
        config.artifact_root,
    )
    return report


def train_select_and_persist(
    config: ModelConfig,
    *,
    tracking_root: str | Path = DEFAULT_TRACKING_ROOT,
    figures_root: str | Path | None = DEFAULT_FIGURES_ROOT,
    validate_expected_counts: bool = True,
) -> dict[str, Any]:
    """Run selection without loading the final test partition until the winner is locked."""
    if config.artifact_root.exists():
        raise ModelTrainingError(
            f"Model output already exists: {config.artifact_root}. Remove it deliberately or choose another output."
        )
    contract = load_feature_contract(config.feature_contract_path)
    manifest = load_dataset_manifest(config.dataset_root)
    validate_dataset_manifest(manifest, contract)
    contract_digest = contract_sha256(config.feature_contract_path)
    train = load_partition(
        config.dataset_root,
        "train",
        contract,
        validate_expected_counts=validate_expected_counts,
    )
    validation = load_partition(
        config.dataset_root,
        "validation",
        contract,
        validate_expected_counts=validate_expected_counts,
    )
    class_ratio = training_class_ratio(train.target)
    configure_tracking(config.experiment_name, tracking_root)
    common_parameters = _common_tracking_parameters(
        config, manifest, contract_digest, class_ratio
    )

    evidence = _train_and_track_candidates(
        train, validation, contract, config, class_ratio, common_parameters
    )

    # This immutable choice is made before test data is loaded. The test partition
    # therefore cannot improve a candidate's rank or change its threshold.
    selected_name = select_candidate(
        {name: item.metrics for name, item in evidence.items()}
    )
    selected = evidence[selected_name]
    LOGGER.info(
        "Validation selection locked: candidate=%s, PR-AUC=%.6f, threshold=%.6f",
        selected_name,
        selected.metrics["pr_auc"],
        selected.threshold.threshold,
    )

    test = load_partition(
        config.dataset_root,
        "test",
        contract,
        validate_expected_counts=validate_expected_counts,
    )
    return _finalize_selected_model(
        config,
        contract,
        manifest,
        contract_digest,
        class_ratio,
        evidence,
        selected_name,
        validation,
        test,
        figures_root,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Expose repository-relative overrides for repeatable local acceptance runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_MODEL_CONFIG_PATH))
    parser.add_argument("--dataset-root")
    parser.add_argument("--artifact-root")
    parser.add_argument("--tracking-root", default=str(DEFAULT_TRACKING_ROOT))
    parser.add_argument("--figures-root", default=str(DEFAULT_FIGURES_ROOT))
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser


def main() -> int:
    """Run the configured workflow and return a shell-friendly status code."""
    arguments = build_argument_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_model_config(arguments.config)
        if arguments.dataset_root:
            config = replace(config, dataset_root=Path(arguments.dataset_root))
        if arguments.artifact_root:
            config = replace(config, artifact_root=Path(arguments.artifact_root))
        report = train_select_and_persist(
            config,
            tracking_root=arguments.tracking_root,
            figures_root=arguments.figures_root,
        )
    except (OSError, ValueError, ModelTrainingError) as error:
        LOGGER.error("Model training failed: %s", error)
        return 1
    LOGGER.info(
        "Model selection completed: candidate=%s, threshold=%.6f, run_id=%s",
        report["selected_model"],
        report["selected_threshold"],
        report["selected_run_id"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
