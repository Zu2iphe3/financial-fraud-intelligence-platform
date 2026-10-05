"""Tests for the verified, read-only model evidence shown in Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dashboard.model_information import (
    ModelInformationError,
    load_model_information,
)


def test_reviewed_model_information_matches_deployed_validation_evidence() -> None:
    information = load_model_information()
    validation = information.validation_evaluation
    test = information.held_out_test_evaluation

    assert information.model_version == "paysim_fraud_model_v2_tuned_review"
    assert information.selected_run_id == "acd17ad55f8e4c2a85d0afedb3608a04"
    assert information.decision_threshold == pytest.approx(0.49207180738449097)
    assert validation.population == "Validation days 19-24 (steps 433-576)"
    assert validation.rows == 192_778
    assert validation.fraud_rows == 1_516
    assert validation.pr_auc == pytest.approx(0.48788556913083986)
    assert validation.roc_auc == pytest.approx(0.9622154668330052)
    assert validation.precision == pytest.approx(0.344041117892708)
    assert validation.recall == pytest.approx(0.7064643799472295)
    assert validation.f1 == pytest.approx(0.4627349319507453)
    assert (
        validation.true_negatives,
        validation.false_positives,
        validation.false_negatives,
        validation.true_positives,
    ) == (189_220, 2_042, 445, 1_071)
    assert test.label == "Held-out test evaluation"
    assert test.population == "Test days 25-31 (steps 577-743)"
    assert test.rows == 161_426
    assert test.fraud_rows == 1_840
    assert test.pr_auc == pytest.approx(0.5398478224767685)
    assert test.roc_auc == pytest.approx(0.9577294735181628)
    assert test.precision == pytest.approx(0.2893252559013996)
    assert test.recall == pytest.approx(0.7527173913043478)
    assert test.f1 == pytest.approx(0.41798702278557415)
    assert (
        test.true_negatives,
        test.false_positives,
        test.false_negatives,
        test.true_positives,
    ) == (156_184, 3_402, 455, 1_385)
    assert validation.threshold == test.threshold == information.decision_threshold
    assert information.v2_test_evaluation_available is True


def test_model_information_rejects_inconsistent_confusion_counts(tmp_path) -> None:
    source = json.loads(
        Path("config/dashboard/fraud_model_information.json").read_text(encoding="utf-8")
    )
    source["held_out_test_evaluation"]["true_positives"] += 1
    path = tmp_path / "model_information.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ModelInformationError, match="Confusion-matrix"):
        load_model_information(path)
