"""Fraud-sensitive probability, threshold, and cohort evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


class EvaluationError(ValueError):
    """Raised when scores or labels cannot support a trustworthy evaluation."""


@dataclass(frozen=True)
class ThresholdSelection:
    """Validation-only operating point and its mandatory fraud trade-off table."""

    threshold: float
    selected_metrics: dict[str, float | int | None]
    comparison: pd.DataFrame
    curve: pd.DataFrame


def _validated_arrays(
    target: Iterable[int], probabilities: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Prevent malformed probabilities from producing plausible-looking metrics."""
    labels = np.asarray(target, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise EvaluationError("Target and probabilities must be aligned one-dimensional arrays")
    if labels.size == 0 or not np.isin(labels, (0, 1)).all():
        raise EvaluationError("Target must contain binary labels")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise EvaluationError("Probabilities must be finite values between zero and one")
    return labels, scores


def evaluate_at_threshold(
    target: Iterable[int], probabilities: Iterable[float], threshold: float
) -> dict[str, float | int | None]:
    """Report all mandatory operating consequences at one decision threshold."""
    labels, scores = _validated_arrays(target, probabilities)
    if not 0 <= threshold <= 1:
        raise EvaluationError("Threshold must be between zero and one")
    predictions = (scores >= threshold).astype(np.int8)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "fraud_precision": float(precision_score(labels, predictions, zero_division=0)),
        "fraud_recall": float(recall_score(labels, predictions, zero_division=0)),
        "fraud_f1": float(f1_score(labels, predictions, zero_division=0)),
        "true_negatives": int(true_negative),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
        "fraud_support": int(labels.sum()),
        "legitimate_support": int((labels == 0).sum()),
        "detected_fraud": int(true_positive),
        "alerts": int(predictions.sum()),
    }


def evaluate_probabilities(
    target: Iterable[int], probabilities: Iterable[float]
) -> dict[str, float | int | None]:
    """Report ranking metrics while making one-class limitations explicit."""
    labels, scores = _validated_arrays(target, probabilities)
    classes = np.unique(labels)
    # PR-AUC is mathematically computable for an all-positive cohort but becomes
    # trivially 1.0, so mark it non-informative instead of overstating performance.
    if len(classes) < 2:
        pr_auc = None
        roc_auc = None
    else:
        pr_auc = float(average_precision_score(labels, scores))
        roc_auc = float(roc_auc_score(labels, scores))
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "fraud_prevalence": float(labels.mean()),
        "fraud_support": int(labels.sum()),
        "rows": int(len(labels)),
    }


def select_validation_threshold(
    target: Iterable[int],
    probabilities: Iterable[float],
    reference_thresholds: Iterable[float],
    reference_recall_targets: Iterable[float] = (),
) -> ThresholdSelection:
    """Maximize validation fraud F1, then recall, then fewer false positives."""
    labels, scores = _validated_arrays(target, probabilities)
    if len(np.unique(labels)) < 2:
        raise EvaluationError("Threshold selection requires both legitimate and fraud rows")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1_values = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    maximum_f1 = float(f1_values.max())
    tied_indexes = np.flatnonzero(np.isclose(f1_values, maximum_f1, rtol=0, atol=1e-12))
    tied_rows = []
    for index in tied_indexes:
        metrics = evaluate_at_threshold(labels, scores, float(thresholds[index]))
        tied_rows.append(metrics)
    selected = sorted(
        tied_rows,
        key=lambda row: (
            -float(row["fraud_recall"]),
            int(row["false_positives"]),
            -float(row["threshold"]),
        ),
    )[0]

    point_labels: dict[float, set[str]] = {}
    for value in reference_thresholds:
        threshold_value = float(value)
        if 0 <= threshold_value <= 1:
            point_labels.setdefault(threshold_value, set()).add(
                f"fixed_threshold_{threshold_value:g}"
            )
    point_labels.setdefault(float(selected["threshold"]), set()).add("maximum_f1")
    for recall_target in reference_recall_targets:
        target_value = float(recall_target)
        if not 0 < target_value <= 1:
            raise EvaluationError("Reference recall targets must be greater than zero and at most one")
        eligible = np.flatnonzero(recall >= target_value)
        if eligible.size:
            best_precision = precision[eligible].max()
            best = eligible[np.flatnonzero(np.isclose(precision[eligible], best_precision))[-1]]
            point_labels.setdefault(float(thresholds[best]), set()).add(
                f"recall_at_least_{target_value:.0%}"
            )
    comparison_rows = []
    for threshold_value, labels_for_point in point_labels.items():
        comparison_rows.append(
            {
                **evaluate_at_threshold(labels, scores, threshold_value),
                "operating_point": ";".join(sorted(labels_for_point)),
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values("threshold", ignore_index=True)
    comparison["selected"] = np.isclose(
        comparison["threshold"], float(selected["threshold"]), rtol=0, atol=1e-12
    )
    curve = pd.DataFrame(
        {
            "threshold": thresholds,
            "fraud_precision": precision,
            "fraud_recall": recall,
            "fraud_f1": f1_values,
        }
    )
    return ThresholdSelection(
        threshold=float(selected["threshold"]),
        selected_metrics=selected,
        comparison=comparison,
        curve=curve,
    )


def combined_evaluation(
    target: Iterable[int], probabilities: Iterable[float], threshold: float
) -> dict[str, Any]:
    """Combine ranking context with mandatory threshold-level decision metrics."""
    return {
        **evaluate_probabilities(target, probabilities),
        **evaluate_at_threshold(target, probabilities, threshold),
    }


def evaluate_daily_top_k(
    target: Iterable[int],
    probabilities: Iterable[float],
    simulation_days: Iterable[int],
    daily_capacities: Iterable[int],
) -> pd.DataFrame:
    """Measure fraud outcomes when analysts review only each day's top scores.

    Stable source-row order resolves equal-score ties so repeated evaluations choose
    exactly the same alerts. Simulation day remains evaluation metadata, not a feature.
    """
    labels, scores = _validated_arrays(target, probabilities)
    days = np.asarray(simulation_days)
    if days.ndim != 1 or days.shape != labels.shape or pd.isna(days).any():
        raise EvaluationError("Simulation days must be present and aligned with scores")
    capacities = tuple(int(value) for value in daily_capacities)
    if not capacities or any(value <= 0 for value in capacities):
        raise EvaluationError("Daily top-k capacities must contain positive integers")
    if len(capacities) != len(set(capacities)):
        raise EvaluationError("Daily top-k capacities must be unique")

    rows: list[dict[str, float | int]] = []
    positions = np.arange(len(labels))
    unique_days = np.unique(days)
    for capacity in capacities:
        predictions = np.zeros(len(labels), dtype=np.int8)
        for day in unique_days:
            day_positions = positions[days == day]
            ranked = day_positions[np.argsort(-scores[day_positions], kind="mergesort")]
            predictions[ranked[:capacity]] = 1
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).ravel()
        alerts = int(predictions.sum())
        rows.append(
            {
                "daily_alert_capacity": capacity,
                "days": int(len(unique_days)),
                "alerts": alerts,
                "average_alerts_per_day": float(alerts / len(unique_days)),
                "fraud_precision": float(precision_score(labels, predictions, zero_division=0)),
                "fraud_recall": float(recall_score(labels, predictions, zero_division=0)),
                "fraud_f1": float(f1_score(labels, predictions, zero_division=0)),
                "true_negatives": int(true_negative),
                "false_positives": int(false_positive),
                "false_negatives": int(false_negative),
                "true_positives": int(true_positive),
                "fraud_support": int(labels.sum()),
            }
        )
    return pd.DataFrame(rows)


def select_candidate(validation_results: dict[str, dict[str, Any]]) -> str:
    """Choose by validation evidence only, with documented fraud-sensitive ties."""
    if not validation_results:
        raise EvaluationError("At least one validation result is required")
    required = {
        "pr_auc",
        "fraud_f1",
        "fraud_recall",
        "false_positives",
    }
    for name, metrics in validation_results.items():
        missing = required - metrics.keys()
        if missing or metrics.get("pr_auc") is None:
            raise EvaluationError(f"Candidate {name!r} has incomplete validation metrics")
        if any(key.startswith("test_") for key in metrics):
            raise EvaluationError("Test metrics must not enter candidate selection")
    return sorted(
        validation_results,
        key=lambda name: (
            -float(validation_results[name]["pr_auc"]),
            -float(validation_results[name]["fraud_f1"]),
            -float(validation_results[name]["fraud_recall"]),
            int(validation_results[name]["false_positives"]),
            name,
        ),
    )[0]
