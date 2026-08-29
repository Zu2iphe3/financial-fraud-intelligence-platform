"""Tests for reproducible PH7-02 validation-only visuals."""

import pandas as pd

from src.models.validation_visuals import create_validation_figures


def _result(scale: float) -> dict:
    return {
        "validation_metrics": {
            "pr_auc": 0.4 * scale,
            "fraud_recall": 0.6 * scale,
            "fraud_f1": 0.5 * scale,
            "threshold": 0.4,
            "true_negatives": 90,
            "false_positives": 10,
            "false_negatives": 4,
            "true_positives": 6,
        },
        "daily_top_k": [
            {"daily_alert_capacity": 100, "fraud_recall": 0.3 * scale, "fraud_precision": 0.7 * scale},
            {"daily_alert_capacity": 500, "fraud_recall": 0.6 * scale, "fraud_precision": 0.3 * scale},
        ],
    }


def test_create_validation_figures_writes_six_review_artifacts(tmp_path):
    baseline = {
        "results": {
            "v1-xgboost": _result(0.7),
            "v2-logistic_regression": _result(0.6),
            "v2-xgboost": _result(0.9),
        }
    }
    search = {"winner": "tuned", "trials": {"tuned": _result(1.0)}}
    curve = pd.DataFrame(
        {"threshold": [0.2, 0.5], "fraud_precision": [0.3, 0.7], "fraud_recall": [0.9, 0.5]}
    )
    importance = pd.DataFrame(
        {"feature": ["amount", "history"], "measure": ["importance"] * 2, "value": [0.3, 0.7]}
    )

    paths = create_validation_figures(
        baseline,
        search,
        {name: curve for name in ("v2-logistic_regression", "v2-xgboost", "tuned-xgboost")},
        importance,
        tmp_path / "figures",
    )

    assert len(paths) == 6
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
