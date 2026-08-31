"""Create validation-only PH7-02 model-review visuals and log them to MLflow."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.features.contract import REPOSITORY_ROOT


# Experiment artifacts are rendered in CI and headless local shells, so they must not
# depend on a desktop GUI or a user-specific Tk installation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)


DEFAULT_BASELINE_REVIEW = (
    REPOSITORY_ROOT / "artifacts/model_reviews/paysim_v2_final_baselines/comparison.json"
)
DEFAULT_SEARCH_REVIEW = (
    REPOSITORY_ROOT / "artifacts/model_reviews/paysim_xgboost_search_v2/search_review.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts/model_reviews/paysim_v2_final_visuals"
DISPLAY_NAMES = {
    "v2-logistic_regression": "Logistic Regression",
    "v2-xgboost": "Untuned XGBoost",
    "tuned-xgboost": "Tuned XGBoost",
}


def _read_json(path: Path) -> dict[str, Any]:
    """Load one local review document with a useful failure boundary."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load validation review {path}: {error}") from error
    if document.get("test_partition_loaded") is not False:
        raise ValueError(f"Review {path} does not prove v2 avoided loading the test partition")
    return document


def _save(figure: plt.Figure, destination: Path) -> None:
    """Write consistent, portfolio-readable PNG artifacts."""
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def create_validation_figures(
    baseline: dict[str, Any],
    search: dict[str, Any],
    curves: dict[str, pd.DataFrame],
    importance: pd.DataFrame,
    output: Path,
) -> tuple[Path, ...]:
    """Render the six approved figures from stored validation evidence only."""
    output.mkdir(parents=True, exist_ok=False)
    winner = search["trials"][search["winner"]]
    models = {
        "v2-logistic_regression": baseline["results"]["v2-logistic_regression"],
        "v2-xgboost": baseline["results"]["v2-xgboost"],
        "tuned-xgboost": winner,
    }
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(8, 6))
    for name, curve in curves.items():
        axis.plot(curve["fraud_recall"], curve["fraud_precision"], label=DISPLAY_NAMES[name])
    axis.set(title="Validation precision-recall curves", xlabel="Fraud recall", ylabel="Fraud precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    paths.append(output / "validation_precision_recall_curve.png")
    _save(figure, paths[-1])

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, (name, result) in zip(axes, models.items()):
        metrics = result["validation_metrics"]
        matrix = np.array(
            [[metrics["true_negatives"], metrics["false_positives"]],
             [metrics["false_negatives"], metrics["true_positives"]]]
        )
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
        axis.set(
            title=DISPLAY_NAMES[name], xlabel="Predicted class", ylabel="Actual class",
            xticks=(0, 1), yticks=(0, 1), xticklabels=("Legitimate", "Fraud"),
            yticklabels=("Legitimate", "Fraud"),
        )
    figure.suptitle("Validation confusion matrices at maximum-F1 review points")
    paths.append(output / "validation_confusion_matrices.png")
    _save(figure, paths[-1])

    tuned_curve = curves["tuned-xgboost"]
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(tuned_curve["threshold"], tuned_curve["fraud_precision"], label="Precision")
    axis.plot(tuned_curve["threshold"], tuned_curve["fraud_recall"], label="Recall")
    axis.axvline(winner["validation_metrics"]["threshold"], color="black", linestyle="--", label="Maximum-F1 review point")
    axis.set(title="Tuned XGBoost validation threshold trade-off", xlabel="Threshold", ylabel="Metric value")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    paths.append(output / "validation_threshold_tradeoff.png")
    _save(figure, paths[-1])

    comparison_names = ("v1-xgboost", "v2-xgboost")
    comparison_metrics = ("pr_auc", "fraud_recall", "fraud_f1")
    values = np.array(
        [[baseline["results"][name]["validation_metrics"][metric] for metric in comparison_metrics]
         for name in comparison_names]
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(comparison_metrics))
    width = 0.36
    axis.bar(positions - width / 2, values[0], width, label="v1 XGBoost")
    axis.bar(positions + width / 2, values[1], width, label="v2 XGBoost")
    axis.set(title="Validation improvement from v1 to behavioural v2", ylabel="Metric value", xticks=positions, xticklabels=("PR-AUC", "Recall", "F1"))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    paths.append(output / "v1_v2_validation_comparison.png")
    _save(figure, paths[-1])

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for name, result in models.items():
        table = pd.DataFrame(result["daily_top_k"])
        axes[0].plot(table["daily_alert_capacity"], table["fraud_recall"], marker="o", label=DISPLAY_NAMES[name])
        axes[1].plot(table["daily_alert_capacity"], table["fraud_precision"], marker="o", label=DISPLAY_NAMES[name])
    axes[0].set(title="Recall by daily review capacity", ylabel="Fraud recall")
    axes[1].set(title="Precision by daily review capacity", ylabel="Fraud precision")
    for axis in axes:
        axis.set(xlabel="Alerts reviewed per day")
        axis.grid(alpha=0.25)
        axis.legend()
    paths.append(output / "validation_daily_top_k.png")
    _save(figure, paths[-1])

    ranked = importance.sort_values("value", ascending=True).tail(15)
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(ranked["feature"], ranked["value"])
    axis.set(title="Tuned XGBoost built-in feature importance", xlabel="Importance")
    axis.grid(axis="x", alpha=0.25)
    paths.append(output / "tuned_xgboost_feature_importance.png")
    _save(figure, paths[-1])
    return tuple(paths)


def generate_and_log_visuals(
    baseline_path: Path = DEFAULT_BASELINE_REVIEW,
    search_path: Path = DEFAULT_SEARCH_REVIEW,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Resolve stored MLflow evidence, create figures, and attach them to the tuned run."""
    baseline = _read_json(baseline_path)
    search = _read_json(search_path)
    winner = search["trials"][search["winner"]]
    run_ids = {
        "v2-logistic_regression": baseline["results"]["v2-logistic_regression"]["run_id"],
        "v2-xgboost": baseline["results"]["v2-xgboost"]["run_id"],
        "tuned-xgboost": winner["run_id"],
    }
    tracking_database = (REPOSITORY_ROOT / "mlruns/mlflow.db").resolve()
    client = MlflowClient(tracking_uri=f"sqlite:///{tracking_database.as_posix()}")
    with tempfile.TemporaryDirectory(prefix="paysim-visuals-") as temporary:
        curves = {
            name: pd.read_csv(
                client.download_artifacts(
                    run_id, "validation/threshold_curve.csv", temporary
                )
            )
            for name, run_id in run_ids.items()
        }
        importance = pd.read_csv(
            client.download_artifacts(
                winner["run_id"], "validation/model_explanation.csv", temporary
            )
        )
        paths = create_validation_figures(baseline, search, curves, importance, output)

    manifest = {
        "selection_partition": "validation",
        "test_partition_loaded": False,
        "production_threshold_selected": False,
        "tuned_run_id": winner["run_id"],
        "figures": [path.name for path in paths],
        "importance_method": "XGBoost built-in feature importance",
        "shap_generated": False,
        "shap_reason": "Built-in importance provides sufficient review evidence without adding another dependency or a costly 6.3M-row explanation pass.",
    }
    manifest_path = output / "visual_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for path in (*paths, manifest_path):
        client.log_artifact(winner["run_id"], str(path), "validation_visuals")
    return manifest


def main() -> int:
    """Generate the validation-only visual review from existing experiment evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    try:
        manifest = generate_and_log_visuals(output=arguments.output)
    except (OSError, ValueError) as error:
        print(f"Validation visual generation failed: {error}")
        return 1
    print(f"Generated {len(manifest['figures'])} validation figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
