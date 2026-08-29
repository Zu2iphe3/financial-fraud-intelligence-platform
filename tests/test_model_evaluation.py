"""Tests for rare-fraud evaluation, thresholding, and candidate selection."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluation import (
    EvaluationError,
    combined_evaluation,
    evaluate_daily_top_k,
    evaluate_probabilities,
    select_candidate,
    select_validation_threshold,
)


def test_daily_top_k_uses_exact_capacity_per_day_and_stable_ties():
    result = evaluate_daily_top_k(
        [1, 0, 0, 1, 0, 0],
        [0.9, 0.8, 0.8, 0.7, 0.6, 0.5],
        [19, 19, 19, 20, 20, 20],
        [1, 2],
    )

    top_one = result.loc[result["daily_alert_capacity"] == 1].iloc[0]
    top_two = result.loc[result["daily_alert_capacity"] == 2].iloc[0]
    assert (top_one["alerts"], top_one["true_positives"], top_one["false_positives"]) == (2, 2, 0)
    assert (top_two["alerts"], top_two["true_positives"], top_two["false_positives"]) == (4, 2, 2)


def test_daily_top_k_rejects_invalid_metadata_or_capacity():
    with pytest.raises(EvaluationError, match="aligned"):
        evaluate_daily_top_k([0, 1], [0.1, 0.9], [19], [1])
    with pytest.raises(EvaluationError, match="positive"):
        evaluate_daily_top_k([0, 1], [0.1, 0.9], [19, 19], [0])


def test_threshold_report_contains_mandatory_decision_metrics():
    labels = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([0.05, 0.2, 0.8, 0.7, 0.9])

    result = combined_evaluation(labels, probabilities, threshold=0.5)

    assert result["fraud_precision"] == pytest.approx(2 / 3)
    assert result["fraud_recall"] == 1.0
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 0
    assert result["true_positives"] == 2
    assert result["alerts"] == 3
    assert result["fraud_support"] == 2
    assert result["pr_auc"] is not None
    assert result["roc_auc"] is not None


def test_threshold_is_selected_from_validation_f1_with_tradeoffs():
    labels = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.7, 0.6, 0.9])

    selection = select_validation_threshold(
        labels, probabilities, (0.25, 0.5, 0.75), (0.5, 0.75, 1.0)
    )

    assert selection.selected_metrics["fraud_f1"] == pytest.approx(0.8)
    assert selection.comparison["selected"].sum() == 1
    assert {
        "fraud_precision",
        "fraud_recall",
        "fraud_f1",
        "false_positives",
        "false_negatives",
    }.issubset(selection.comparison.columns)
    assert selection.comparison["operating_point"].str.contains("recall_at_least").any()


def test_candidate_selection_uses_pr_auc_before_accuracy_or_roc_auc():
    results = {
        "accurate_but_weaker": {
            "pr_auc": 0.40,
            "fraud_f1": 0.80,
            "fraud_recall": 0.90,
            "false_positives": 1,
            "accuracy": 0.999,
            "roc_auc": 0.99,
        },
        "stronger_pr_ranking": {
            "pr_auc": 0.50,
            "fraud_f1": 0.70,
            "fraud_recall": 0.80,
            "false_positives": 5,
            "accuracy": 0.90,
            "roc_auc": 0.80,
        },
    }

    assert select_candidate(results) == "stronger_pr_ranking"


def test_test_metrics_are_rejected_from_selection():
    with pytest.raises(EvaluationError, match="Test metrics"):
        select_candidate(
            {
                "candidate": {
                    "pr_auc": 0.5,
                    "fraud_f1": 0.4,
                    "fraud_recall": 0.3,
                    "false_positives": 2,
                    "test_pr_auc": 0.9,
                }
            }
        )


def test_day_31_one_class_metrics_are_marked_non_informative():
    result = evaluate_probabilities([1, 1, 1], [0.2, 0.8, 0.9])

    assert result["pr_auc"] is None
    assert result["roc_auc"] is None
    assert result["fraud_prevalence"] == 1.0


def test_threshold_selection_rejects_one_class_target():
    with pytest.raises(EvaluationError, match="both legitimate and fraud"):
        select_validation_threshold([1, 1], [0.2, 0.8], (0.5,))
