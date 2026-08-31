"""Tests for the bounded, validation-only XGBoost tuning workflow."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.models.candidates import TrainedCandidate
from src.models.config import load_model_config
from src.models.data import PartitionData
from src.models.tune_xgboost import load_search_config, run_xgboost_search


def _partition(labels: list[int]) -> PartitionData:
    rows = len(labels)
    return PartitionData(
        features=pd.DataFrame(
            {
                "amount_log1p": pd.Series(np.linspace(1, 5, rows), dtype="float32"),
                "transaction_type_cash_in": pd.Series([0] * rows, dtype="int8"),
                "transaction_type_cash_out": pd.Series([1] * rows, dtype="int8"),
                "transaction_type_debit": pd.Series([0] * rows, dtype="int8"),
                "transaction_type_payment": pd.Series([0] * rows, dtype="int8"),
                "transaction_type_transfer": pd.Series([0] * rows, dtype="int8"),
            }
        ),
        target=pd.Series(labels, dtype="int8", name="is_fraud"),
        metadata=pd.DataFrame({"simulation_day": [19] * rows}),
    )


def test_search_config_is_bounded_and_includes_requested_recall_targets():
    search = load_search_config()

    assert search.primary_metric == "validation_pr_auc"
    assert len(search.trials) == 9
    assert len({trial.name for trial in search.trials}) == 9
    assert search.reference_recall_targets == (0.5, 0.75, 0.8, 0.9)
    assert search.daily_top_k_capacities == (100, 500, 1000, 2000)
    assert {trial.scale_pos_weight for trial in search.trials} >= {
        1.0, 5.0, 10.0, 25.0, 35.0, 50.0
    }


def test_v2_search_is_one_bounded_validation_only_experiment():
    search = load_search_config("config/models/paysim_xgboost_search_v2.json")

    assert search.search_version == "paysim_xgboost_search_v2"
    assert search.primary_metric == "validation_pr_auc"
    assert len(search.trials) == 12
    assert search.reference_recall_targets == (0.5, 0.75, 0.8, 0.9)
    assert search.daily_top_k_capacities == (100, 500, 1000, 2000)


def test_search_never_loads_test_partition(monkeypatch, tmp_path):
    train = _partition([0, 0, 0, 1])
    validation = _partition([0, 0, 1, 1])
    loaded: list[str] = []

    def fake_load_partition(dataset_root, partition, contract):
        loaded.append(partition)
        return train if partition == "train" else validation

    monkeypatch.setattr("src.models.tune_xgboost.load_partition", fake_load_partition)
    monkeypatch.setattr("src.models.tune_xgboost.load_dataset_manifest", lambda root: {
        "contract_version": "paysim_features_v1",
        "source_dataset_id": "a" * 64,
    })
    monkeypatch.setattr("src.models.tune_xgboost.validate_dataset_manifest", lambda *args: None)
    monkeypatch.setattr("src.models.tune_xgboost.configure_tracking", lambda *args: None)
    monkeypatch.setattr(
        "src.models.tune_xgboost.train_candidate",
        lambda *args, **kwargs: TrainedCandidate("xgboost", object(), {"best_iteration": 1}),
    )
    monkeypatch.setattr(
        "src.models.tune_xgboost.fraud_probabilities",
        lambda candidate, features: np.array([0.1, 0.2, 0.8, 0.9]),
    )
    monkeypatch.setattr(
        "src.models.tune_xgboost.log_candidate_run",
        lambda *args, run_name=None, extra_tags=None, **kwargs: f"run-{run_name}",
    )
    search = load_search_config()
    search = replace(search, trials=search.trials[:2])

    result = run_xgboost_search(
        load_model_config(),
        search,
        tracking_root=tmp_path / "tracking",
        review_root=tmp_path / "review",
    )

    assert loaded == ["train", "validation"]
    assert result["test_partition_loaded"] is False
    assert len(result["results"]) == 2
