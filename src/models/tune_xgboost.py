"""Run a bounded XGBoost search using only PH7-01 train and validation data."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.contract import REPOSITORY_ROOT, load_feature_contract
from src.models.candidates import fraud_probabilities, train_candidate
from src.models.config import DEFAULT_MODEL_CONFIG_PATH, ModelConfig, load_model_config
from src.models.data import (
    contract_sha256,
    load_dataset_manifest,
    load_partition,
    training_class_ratio,
    validate_dataset_manifest,
)
from src.models.evaluation import (
    combined_evaluation,
    evaluate_daily_top_k,
    select_candidate,
    select_validation_threshold,
)
from src.models.tracking import DEFAULT_TRACKING_ROOT, configure_tracking, log_candidate_run


LOGGER = logging.getLogger(__name__)
DEFAULT_SEARCH_PATH = (
    REPOSITORY_ROOT / "config" / "models" / "paysim_xgboost_final_baseline_v1.json"
)
DEFAULT_REVIEW_ROOT = REPOSITORY_ROOT / "artifacts" / "model_reviews"
_TRIAL_PARAMETERS = {
    "max_depth",
    "min_child_weight",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
}


class XGBoostSearchError(ValueError):
    """Raised when a tuning review could use test data or cannot be reproduced."""


@dataclass(frozen=True)
class SearchTrial:
    """One deliberately chosen parameter configuration in the bounded search."""

    name: str
    parameters: dict[str, float | int]
    scale_pos_weight: float | None
    scale_pos_weight_multiplier: float | None


@dataclass(frozen=True)
class SearchConfig:
    """Validation-only optimization settings and candidate definitions."""

    search_version: str
    experiment_name: str
    primary_metric: str
    n_estimators: int
    early_stopping_rounds: int
    n_jobs: int
    reference_recall_targets: tuple[float, ...]
    daily_top_k_capacities: tuple[int, ...]
    trials: tuple[SearchTrial, ...]


def load_search_config(path: str | Path = DEFAULT_SEARCH_PATH) -> SearchConfig:
    """Load the committed bounded search and reject accidental scope expansion."""
    search_path = Path(path)
    if not search_path.is_absolute():
        search_path = REPOSITORY_ROOT / search_path
    try:
        document = json.loads(search_path.read_text(encoding="utf-8"))
        trials = []
        for item in document["trials"]:
            unexpected = set(item) - {
                "name",
                "scale_pos_weight",
                "scale_pos_weight_multiplier",
            } - _TRIAL_PARAMETERS
            if unexpected:
                raise XGBoostSearchError(
                    f"Trial {item.get('name')!r} contains unsupported parameters: {sorted(unexpected)}"
                )
            trials.append(
                SearchTrial(
                    name=str(item["name"]),
                    parameters={name: item[name] for name in _TRIAL_PARAMETERS},
                    scale_pos_weight=(
                        float(item["scale_pos_weight"])
                        if "scale_pos_weight" in item
                        else None
                    ),
                    scale_pos_weight_multiplier=(
                        float(item["scale_pos_weight_multiplier"])
                        if "scale_pos_weight_multiplier" in item
                        else None
                    ),
                )
            )
        config = SearchConfig(
            search_version=str(document["search_version"]),
            experiment_name=str(document["experiment_name"]),
            primary_metric=str(document["primary_metric"]),
            n_estimators=int(document["n_estimators"]),
            early_stopping_rounds=int(document["early_stopping_rounds"]),
            n_jobs=int(document["n_jobs"]),
            reference_recall_targets=tuple(
                float(value) for value in document["reference_recall_targets"]
            ),
            daily_top_k_capacities=tuple(
                int(value) for value in document.get("daily_top_k_capacities", [])
            ),
            trials=tuple(trials),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, XGBoostSearchError):
            raise
        raise XGBoostSearchError(f"Cannot load XGBoost search {search_path}: {error}") from error

    names = [trial.name for trial in config.trials]
    if config.primary_metric != "validation_pr_auc":
        raise XGBoostSearchError("The search must optimize validation PR-AUC")
    if not 1 <= len(config.trials) <= 20 or len(names) != len(set(names)):
        raise XGBoostSearchError("The bounded search requires 1-20 uniquely named trials")
    for trial in config.trials:
        weights = (trial.scale_pos_weight, trial.scale_pos_weight_multiplier)
        if sum(value is not None for value in weights) != 1:
            raise XGBoostSearchError(
                f"Trial {trial.name!r} must define exactly one class-weight strategy"
            )
        if any(value is not None and value <= 0 for value in weights):
            raise XGBoostSearchError("Class weights and multipliers must be positive")
    if config.daily_top_k_capacities and (
        any(value <= 0 for value in config.daily_top_k_capacities)
        or len(config.daily_top_k_capacities) != len(set(config.daily_top_k_capacities))
    ):
        raise XGBoostSearchError("Daily top-k capacities must be positive and unique")
    return config


def _trial_model_config(
    model_config: ModelConfig,
    search: SearchConfig,
    trial: SearchTrial,
) -> ModelConfig:
    """Apply only reviewed XGBoost parameters while preserving every other contract."""
    parameters = {
        **dict(model_config.xgboost),
        **trial.parameters,
        "n_estimators": search.n_estimators,
        "early_stopping_rounds": search.early_stopping_rounds,
        "n_jobs": search.n_jobs,
    }
    return replace(
        model_config,
        experiment_name=search.experiment_name,
        xgboost=parameters,
        reference_recall_targets=search.reference_recall_targets,
    )


def _write_search_review(
    destination: Path,
    search: SearchConfig,
    results: dict[str, dict[str, Any]],
    winner: str,
) -> None:
    """Persist ignored validation evidence without creating a selected test artifact."""
    if destination.exists():
        raise XGBoostSearchError(
            f"Search review already exists: {destination}. Remove it deliberately before rerunning."
        )
    destination.mkdir(parents=True)
    review = {
        "search_version": search.search_version,
        "primary_metric": search.primary_metric,
        "selection_partition": "validation",
        "test_partition_loaded": False,
        "winner": winner,
        "trials": results,
    }
    (destination / "search_review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    leaderboard = pd.DataFrame(
        [
            {
                "trial": name,
                "run_id": result["run_id"],
                **result["parameters"],
                **result["validation_metrics"],
            }
            for name, result in results.items()
        ]
    ).sort_values(
        ["pr_auc", "fraud_f1", "fraud_recall", "false_positives"],
        ascending=[False, False, False, True],
        ignore_index=True,
    )
    leaderboard.to_csv(destination / "validation_leaderboard.csv", index=False)
    pd.DataFrame(results[winner]["threshold_operating_points"]).to_csv(
        destination / "winner_thresholds.csv", index=False
    )
    if results[winner]["daily_top_k"]:
        pd.DataFrame(results[winner]["daily_top_k"]).to_csv(
            destination / "winner_top_k.csv", index=False
        )


def run_xgboost_search(
    model_config: ModelConfig,
    search: SearchConfig,
    *,
    tracking_root: str | Path = DEFAULT_TRACKING_ROOT,
    review_root: str | Path = DEFAULT_REVIEW_ROOT,
) -> dict[str, Any]:
    """Optimize on validation PR-AUC without loading or evaluating the test split."""
    contract = load_feature_contract(model_config.feature_contract_path)
    manifest = load_dataset_manifest(model_config.dataset_root)
    validate_dataset_manifest(manifest, contract)
    train = load_partition(model_config.dataset_root, "train", contract)
    validation = load_partition(model_config.dataset_root, "validation", contract)
    class_ratio = training_class_ratio(train.target)
    configure_tracking(search.experiment_name, tracking_root)
    common = {
        "search_version": search.search_version,
        "feature_contract_version": contract.contract_version,
        "feature_contract_sha256": contract_sha256(model_config.feature_contract_path),
        "source_dataset_id": manifest["source_dataset_id"],
        "random_seed": model_config.random_seed,
        "primary_selection_metric": search.primary_metric,
        "train_rows": len(train.target),
        "validation_rows": len(validation.target),
        "test_partition_loaded": False,
    }

    results: dict[str, dict[str, Any]] = {}
    for index, trial in enumerate(search.trials, start=1):
        LOGGER.info("Tuning XGBoost trial %s/%s: %s", index, len(search.trials), trial.name)
        trial_config = _trial_model_config(model_config, search, trial)
        scale_pos_weight = (
            trial.scale_pos_weight
            if trial.scale_pos_weight is not None
            else class_ratio * float(trial.scale_pos_weight_multiplier)
        )
        candidate = train_candidate(
            "xgboost",
            train.features,
            train.target,
            validation.features,
            validation.target,
            trial_config,
            scale_pos_weight,
        )
        probabilities = fraud_probabilities(candidate, validation.features)
        threshold = select_validation_threshold(
            validation.target,
            probabilities,
            trial_config.reference_thresholds,
            search.reference_recall_targets,
        )
        metrics = combined_evaluation(
            validation.target, probabilities, threshold.threshold
        )
        top_k = (
            evaluate_daily_top_k(
                validation.target,
                probabilities,
                validation.metadata["simulation_day"],
                search.daily_top_k_capacities,
            )
            if search.daily_top_k_capacities
            else pd.DataFrame()
        )
        top_k_metrics = {
            f"validation_top_{int(row.daily_alert_capacity)}_{metric}": getattr(row, metric)
            for row in top_k.itertuples(index=False)
            for metric in (
                "fraud_precision",
                "fraud_recall",
                "fraud_f1",
                "false_positives",
                "false_negatives",
                "true_positives",
                "alerts",
            )
        }
        run_id = log_candidate_run(
            candidate,
            metrics,
            threshold,
            contract.feature_names,
            {
                **common,
                "trial_number": index,
                "scale_pos_weight_strategy": (
                    "absolute" if trial.scale_pos_weight is not None else "ratio_multiplier"
                ),
            },
            run_name=f"{index:02d}-{trial.name}",
            extra_tags={
                "tuning_trial": "true",
                "trial_name": trial.name,
                "search_version": search.search_version,
            },
            extra_metrics=top_k_metrics,
            extra_tables={"daily_top_k.csv": top_k} if not top_k.empty else None,
        )
        results[trial.name] = {
            "run_id": run_id,
            "parameters": candidate.tracked_parameters,
            "validation_metrics": metrics,
            "threshold_operating_points": threshold.comparison.to_dict(orient="records"),
            "daily_top_k": top_k.to_dict(orient="records"),
        }
        del candidate, probabilities
        gc.collect()

    winner = select_candidate(
        {name: result["validation_metrics"] for name, result in results.items()}
    )
    destination = Path(review_root) / search.search_version
    _write_search_review(destination, search, results, winner)
    return {
        "winner": winner,
        "winner_result": results[winner],
        "results": results,
        "review_directory": str(destination),
        "test_partition_loaded": False,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Build repository-relative options for the validation-only search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG_PATH))
    parser.add_argument("--search-config", default=str(DEFAULT_SEARCH_PATH))
    parser.add_argument("--tracking-root", default=str(DEFAULT_TRACKING_ROOT))
    parser.add_argument("--review-root", default=str(DEFAULT_REVIEW_ROOT))
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser


def main() -> int:
    """Run the bounded search and return a shell-friendly status code."""
    arguments = build_argument_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        result = run_xgboost_search(
            load_model_config(arguments.model_config),
            load_search_config(arguments.search_config),
            tracking_root=arguments.tracking_root,
            review_root=arguments.review_root,
        )
    except (OSError, ValueError) as error:
        LOGGER.error("XGBoost tuning failed: %s", error)
        return 1
    winner = result["winner_result"]
    LOGGER.info(
        "Validation-only tuning completed: winner=%s, PR-AUC=%.6f, threshold=%.6f",
        result["winner"],
        winner["validation_metrics"]["pr_auc"],
        winner["validation_metrics"]["threshold"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
