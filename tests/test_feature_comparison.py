"""Tests for the validation-only v1/v2 comparison boundary."""

import numpy as np
import pandas as pd

from src.features.contract import load_feature_contract
from src.models.candidates import TrainedCandidate
from src.models.data import PartitionData
from src.models.compare_paysim_features import run_validation_comparison


def test_feature_comparison_never_loads_test(monkeypatch, tmp_path):
    contract = load_feature_contract("config/features/paysim_v2.json")
    features = pd.DataFrame(
        {name: pd.Series(
            [0, 0, 1, 1] if definition.dtype == "int8" else np.linspace(0, 1, 4),
            dtype=definition.dtype,
        )
         for name, definition in zip(contract.feature_names, contract.features)}
    )
    partition = PartitionData(
        features=features,
        target=pd.Series([0, 0, 1, 1], dtype="int8"),
        metadata=pd.DataFrame({"simulation_day": [19] * 4}),
    )
    loaded: list[str] = []

    def fake_load(root, name, feature_contract):
        loaded.append(name)
        return partition

    monkeypatch.setattr("src.models.compare_paysim_features.load_partition", fake_load)
    monkeypatch.setattr("src.models.compare_paysim_features.load_dataset_manifest", lambda root: {})
    monkeypatch.setattr("src.models.compare_paysim_features.validate_dataset_manifest", lambda *args: None)
    monkeypatch.setattr("src.models.compare_paysim_features.configure_tracking", lambda *args: None)
    monkeypatch.setattr(
        "src.models.compare_paysim_features.train_candidate",
        lambda name, *args, **kwargs: TrainedCandidate(name, object(), {}),
    )
    monkeypatch.setattr(
        "src.models.compare_paysim_features.fraud_probabilities",
        lambda *args: np.array([0.1, 0.2, 0.8, 0.9]),
    )
    monkeypatch.setattr(
        "src.models.compare_paysim_features.log_candidate_run",
        lambda candidate, *args, run_name=None, **kwargs: f"run-{run_name}",
    )
    monkeypatch.setattr("src.models.compare_paysim_features.REVIEW_ROOT", tmp_path / "review")
    monkeypatch.setattr(
        "src.models.compare_paysim_features.DATASETS",
        {"v2": ("config/features/paysim_v2.json", tmp_path / "dataset")},
    )

    result = run_validation_comparison()

    assert loaded == ["train", "validation"]
    assert result["test_partition_loaded"] is False
    assert len(result["results"]) == 3
