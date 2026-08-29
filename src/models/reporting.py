"""Create compact, portfolio-ready evidence from model evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve


def plot_validation_precision_recall(
    target: pd.Series,
    probabilities: Mapping[str, np.ndarray],
    output_path: str | Path,
) -> None:
    """Compare candidate ranking on the one shared validation period."""
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(9, 6))
    for name, scores in probabilities.items():
        precision, recall, _ = precision_recall_curve(target, scores)
        axis.plot(recall, precision, label=name.replace("_", " ").title())
    axis.axhline(float(target.mean()), color="grey", linestyle="--", label="Fraud prevalence")
    axis.set(
        title="Validation precision-recall comparison",
        xlabel="Fraud recall",
        ylabel="Fraud precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(loc="best")
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_threshold_tradeoff(curve: pd.DataFrame, threshold: float, output_path: str | Path) -> None:
    """Show why the recommended F1 threshold is only one possible operating point."""
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(9, 6))
    for metric in ("fraud_precision", "fraud_recall", "fraud_f1"):
        axis.plot(curve["threshold"], curve[metric], label=metric.replace("fraud_", "").upper())
    axis.axvline(threshold, color="black", linestyle="--", label=f"Selected {threshold:.4f}")
    axis.set(
        title="Selected model validation threshold trade-off",
        xlabel="Probability threshold",
        ylabel="Metric value",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.set_xticks(np.linspace(0, 1, 6))
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.tick_params(labelbottom=True, labelleft=True)
    axis.legend(loc="best")
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_test_confusion_matrices(
    cohort_metrics: Mapping[str, Mapping[str, Any]], output_path: str | Path
) -> None:
    """Keep the full, standard, and one-class artifact cohorts visibly separate."""
    sns.set_theme(style="white")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, (name, metrics) in zip(axes, cohort_metrics.items()):
        matrix = np.array(
            [
                [metrics["true_negatives"], metrics["false_positives"]],
                [metrics["false_negatives"], metrics["true_positives"]],
            ]
        )
        sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axis)
        axis.set(
            title=name.replace("_", " ").title(),
            xlabel="Predicted label",
            ylabel="Actual label",
            xticklabels=("Legitimate", "Fraud"),
            yticklabels=("Legitimate", "Fraud"),
        )
    figure.suptitle("Selected model test-cohort confusion matrices")
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
