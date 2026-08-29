"""Tests for leakage-safe loading of generated PH7-01 model inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.features.contract import load_feature_contract
from src.models.config import load_model_config
from src.models.data import (
    ModelDataError,
    load_partition,
    training_class_ratio,
    validate_dataset_manifest,
)


def _manifest_from_contract():
    contract = load_feature_contract()
    return contract, {
        "contract_version": contract.contract_version,
        "source_dataset_id": "a" * 64,
        "feature_schema": [
            {"name": feature.name, "dtype": feature.dtype} for feature in contract.features
        ],
        "target": contract.target_column,
        "splits": {
            split.name: {
                "first_step": split.first_step,
                "last_step": split.last_step,
                "rows": split.expected_rows,
                "fraud_rows": split.expected_fraud_rows,
            }
            for split in contract.splits
        },
        "final_day_policy": {
            "simulation_day": contract.final_day.simulation_day,
            "first_step": contract.final_day.first_step,
            "last_step": contract.final_day.last_step,
            "evaluation_cohort": contract.final_day.evaluation_cohort,
            "predictive_feature": False,
        },
    }


def _write_partition(root: Path, *, include_leakage: bool = False) -> None:
    contract = load_feature_contract()
    features = pd.DataFrame(
        {
            "amount_log1p": pd.Series([1.0, 2.0], dtype="float32"),
            "transaction_type_cash_in": pd.Series([1, 0], dtype="int8"),
            "transaction_type_cash_out": pd.Series([0, 1], dtype="int8"),
            "transaction_type_debit": pd.Series([0, 0], dtype="int8"),
            "transaction_type_payment": pd.Series([0, 0], dtype="int8"),
            "transaction_type_transfer": pd.Series([0, 0], dtype="int8"),
        }
    )
    if include_leakage:
        features["old_balance_orig"] = 0.0
    target = pd.DataFrame({contract.target_column: pd.Series([0, 1], dtype="int8")})
    metadata = pd.DataFrame(
        {
            "transaction_id": ["one", "two"],
            "source_dataset_id": ["a" * 64] * 2,
            "source_row_number": [1, 2],
            "step": [1, 2],
            "simulation_day": [1, 1],
            "partition": ["train", "train"],
            "evaluation_cohort": ["standard_period", "standard_period"],
        }
    )
    for group, frame in (("features", features), ("target", target), ("metadata", metadata)):
        path = root / "train" / group
        path.mkdir(parents=True)
        frame.to_parquet(path / "part-00001.parquet", index=False)


def test_default_model_config_protects_selection_policy():
    config = load_model_config()

    assert config.primary_selection_metric == "validation_pr_auc"
    assert config.threshold_objective == "fraud_f1"
    assert config.threshold_tie_breakers == (
        "fraud_recall",
        "false_positives",
        "threshold",
    )
    assert config.reference_recall_targets == (0.5, 0.75, 0.8, 0.9, 0.95)


def test_manifest_reconciles_with_feature_contract():
    contract, manifest = _manifest_from_contract()

    validate_dataset_manifest(manifest, contract)


def test_manifest_rejects_changed_split_counts():
    contract, manifest = _manifest_from_contract()
    manifest["splits"]["train"]["fraud_rows"] += 1

    with pytest.raises(ModelDataError, match="train reconciliation"):
        validate_dataset_manifest(manifest, contract)


def test_partition_loader_preserves_schema_and_alignment(tmp_path):
    contract = load_feature_contract()
    _write_partition(tmp_path)

    partition = load_partition(tmp_path, "train", contract, validate_expected_counts=False)

    assert tuple(partition.features.columns) == contract.feature_names
    assert partition.target.tolist() == [0, 1]
    assert partition.metadata["transaction_id"].tolist() == ["one", "two"]


def test_partition_loader_rejects_prohibited_feature(tmp_path):
    contract = load_feature_contract()
    _write_partition(tmp_path, include_leakage=True)

    with pytest.raises(ModelDataError, match="feature order"):
        load_partition(tmp_path, "train", contract, validate_expected_counts=False)


def test_training_class_ratio_uses_both_classes():
    target = pd.Series([0, 0, 0, 1], dtype="int8")

    assert training_class_ratio(target) == 3.0

    with pytest.raises(ModelDataError, match="both legitimate and fraud"):
        training_class_ratio(pd.Series([0, 0], dtype="int8"))
