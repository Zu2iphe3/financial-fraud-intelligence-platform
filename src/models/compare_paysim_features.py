"""Compare finalized v1 and v2 baselines using validation data only."""

from __future__ import annotations

import gc
import json
import logging

import pandas as pd

from src.features.contract import REPOSITORY_ROOT, load_feature_contract
from src.models.candidates import fraud_probabilities, train_candidate
from src.models.config import load_model_config
from src.models.data import load_dataset_manifest, load_partition, validate_dataset_manifest
from src.models.evaluation import (
    combined_evaluation,
    evaluate_daily_top_k,
    select_candidate,
    select_validation_threshold,
)
from src.models.tracking import configure_tracking, log_candidate_run


LOGGER = logging.getLogger(__name__)
CONFIG_PATH = REPOSITORY_ROOT / "config/models/paysim_fraud_v2.json"
REVIEW_ROOT = REPOSITORY_ROOT / "artifacts/model_reviews/paysim_v2_final_baselines"
DAILY_CAPACITIES = (100, 500, 1000, 2000)
DATASETS = {
    "v1": (
        REPOSITORY_ROOT / "config/features/paysim_v1.json",
        REPOSITORY_ROOT / "data/processed/paysim_features/paysim_features_v1",
    ),
    "v2": (
        REPOSITORY_ROOT / "config/features/paysim_v2.json",
        REPOSITORY_ROOT / "data/processed/paysim_features/paysim_features_v2",
    ),
}


def _flatten_top_k(table: pd.DataFrame) -> dict[str, float | int]:
    """Give MLflow searchable top-k metrics while preserving the complete table."""
    return {
        f"validation_top_{int(row.daily_alert_capacity)}_{metric}": getattr(row, metric)
        for row in table.itertuples(index=False)
        for metric in (
            "fraud_precision", "fraud_recall", "fraud_f1", "false_positives",
            "false_negatives", "true_positives", "alerts",
        )
    }


def run_validation_comparison() -> dict:
    """Run identical candidates on v1/v2 without ever loading their test partitions."""
    if REVIEW_ROOT.exists():
        raise ValueError(f"Review output already exists: {REVIEW_ROOT}")
    config = load_model_config(CONFIG_PATH)
    configure_tracking(config.experiment_name)
    results: dict[str, dict] = {}

    for variant, (contract_path, dataset_root) in DATASETS.items():
        contract = load_feature_contract(contract_path)
        manifest = load_dataset_manifest(dataset_root)
        validate_dataset_manifest(manifest, contract)
        train = load_partition(dataset_root, "train", contract)
        validation = load_partition(dataset_root, "validation", contract)
        for candidate_name in ("no_skill", "logistic_regression", "xgboost"):
            run_name = f"{variant}-{candidate_name}"
            LOGGER.info("Training %s", run_name)
            candidate = train_candidate(
                candidate_name,
                train.features,
                train.target,
                validation.features,
                validation.target,
                config,
                100.0,
            )
            probabilities = fraud_probabilities(candidate, validation.features)
            threshold = select_validation_threshold(
                validation.target,
                probabilities,
                config.reference_thresholds,
                config.reference_recall_targets,
            )
            metrics = combined_evaluation(
                validation.target, probabilities, threshold.threshold
            )
            top_k = evaluate_daily_top_k(
                validation.target,
                probabilities,
                validation.metadata["simulation_day"],
                DAILY_CAPACITIES,
            )
            run_id = log_candidate_run(
                candidate,
                metrics,
                threshold,
                contract.feature_names,
                {
                    "feature_contract_version": contract.contract_version,
                    "feature_variant": variant,
                    "test_partition_loaded": False,
                    "xgboost_scale_pos_weight": 100.0,
                },
                run_name=run_name,
                extra_tags={"validation_review": "true", "feature_variant": variant},
                extra_metrics=_flatten_top_k(top_k),
                extra_tables={"daily_top_k.csv": top_k},
            )
            results[run_name] = {
                "run_id": run_id,
                "feature_variant": variant,
                "candidate": candidate_name,
                "feature_names": list(contract.feature_names),
                "parameters": candidate.tracked_parameters,
                "validation_metrics": metrics,
                "thresholds": threshold.comparison.to_dict(orient="records"),
                "daily_top_k": top_k.to_dict(orient="records"),
            }
            del candidate, probabilities
            gc.collect()
        del train, validation
        gc.collect()

    winner = select_candidate(
        {name: result["validation_metrics"] for name, result in results.items()}
    )
    REVIEW_ROOT.mkdir(parents=True)
    review = {
        "selection_partition": "validation",
        "test_partition_loaded": False,
        "winner": winner,
        "results": results,
    }
    (REVIEW_ROOT / "comparison.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [{"run": name, **result["validation_metrics"]} for name, result in results.items()]
    ).sort_values("pr_auc", ascending=False).to_csv(
        REVIEW_ROOT / "validation_leaderboard.csv", index=False
    )
    for name, result in results.items():
        pd.DataFrame(result["daily_top_k"]).to_csv(
            REVIEW_ROOT / f"{name}_top_k.csv", index=False
        )
        pd.DataFrame(result["thresholds"]).to_csv(
            REVIEW_ROOT / f"{name}_thresholds.csv", index=False
        )
    return review


def main() -> int:
    """Run the validation-only feature comparison."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        result = run_validation_comparison()
    except (OSError, ValueError) as error:
        LOGGER.error("Feature comparison failed: %s", error)
        return 1
    LOGGER.info("Validation-only comparison completed: winner=%s", result["winner"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
