"""Tests for immutable held-out evaluation safeguards and cohort reporting."""

from __future__ import annotations

import numpy as np
import pytest

from src.api.config import load_api_settings
from src.models.evaluate_frozen_model import (
    FrozenModelEvaluationError,
    _evaluate_cohorts,
    _verify_frozen_identity,
    evaluate_frozen_model,
)


class _Bundle:
    def __init__(self, identity) -> None:
        self.manifest_identity = identity


def _selection_evidence(settings, *, test_loaded=False):
    return {
        "winner": "high_child_weight",
        "selection_partition": "validation",
        "test_partition_loaded": test_loaded,
        "trials": {
            "high_child_weight": {
                "run_id": settings.selected_run_id,
                "validation_metrics": {
                    "threshold": settings.risk_policy.high_min_probability
                },
            }
        },
    }


def test_selection_evidence_proves_test_isolation_before_evaluation() -> None:
    settings = load_api_settings()
    bundle = _Bundle(
        (
            settings.model_version,
            settings.feature_contract_version,
            settings.feature_contract_sha256,
            settings.feature_names,
            settings.selected_model,
            settings.selected_run_id,
        )
    )
    visuals = {
        "selection_partition": "validation",
        "test_partition_loaded": False,
        "tuned_run_id": settings.selected_run_id,
    }

    result = _verify_frozen_identity(
        settings,
        bundle,
        _selection_evidence(settings),
        visuals,
        (
            {"selection_partition": "validation", "test_partition_loaded": False},
            {"selection_partition": "validation", "test_partition_loaded": False},
        ),
    )

    assert result["model_fitting_used_test"] is False
    assert result["hyperparameter_selection_used_test"] is False
    assert result["feature_selection_used_test"] is False
    assert result["threshold_selection_used_test"] is False

    with pytest.raises(FrozenModelEvaluationError, match="frozen validation choice"):
        _verify_frozen_identity(
            settings,
            bundle,
            _selection_evidence(settings, test_loaded=True),
            visuals,
            (
                {"selection_partition": "validation", "test_partition_loaded": False},
                {"selection_partition": "validation", "test_partition_loaded": False},
            ),
        )


def test_predefined_test_cohorts_use_one_unchanged_threshold() -> None:
    target = np.array([0, 1, 0, 1, 1], dtype=np.int8)
    probabilities = np.array([0.1, 0.8, 0.7, 0.9, 0.4])
    cohorts = np.array(
        ["standard_period", "standard_period", "standard_period", "final_day_artifact", "final_day_artifact"]
    )

    result = _evaluate_cohorts(target, probabilities, cohorts, threshold=0.5)

    assert set(result) == {
        "full_test",
        "standard_period_days_25_30",
        "day_31_artifact",
    }
    assert all(metrics["threshold"] == 0.5 for metrics in result.values())
    assert result["day_31_artifact"]["pr_auc"] is None
    assert result["day_31_artifact"]["roc_auc"] is None


def test_final_evaluation_refuses_to_overwrite_existing_report(tmp_path) -> None:
    output = tmp_path / "evaluation.json"
    output.write_text("{}", encoding="utf-8")

    with pytest.raises(FrozenModelEvaluationError, match="already exists"):
        evaluate_frozen_model(output)
