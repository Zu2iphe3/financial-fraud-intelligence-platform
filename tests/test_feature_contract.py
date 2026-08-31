"""Tests for the versioned leakage-safe PaySim feature contract."""

import copy
import json
from pathlib import Path

import pytest

from src.features.contract import (
    DEFAULT_CONTRACT_PATH,
    FeatureContractError,
    load_feature_contract,
    parse_feature_contract,
)


def _document():
    return json.loads(DEFAULT_CONTRACT_PATH.read_text(encoding="utf-8"))


def test_committed_contract_has_stable_minimal_feature_schema():
    contract = load_feature_contract()

    assert contract.contract_version == "paysim_features_v1"
    assert contract.predictive_source_columns == ("amount", "transaction_type")
    assert contract.feature_names == (
        "amount_log1p",
        "transaction_type_cash_in",
        "transaction_type_cash_out",
        "transaction_type_debit",
        "transaction_type_payment",
        "transaction_type_transfer",
    )
    assert tuple(split.name for split in contract.splits) == (
        "train",
        "validation",
        "test",
    )
    assert [
        (
            split.first_step,
            split.last_step,
            split.expected_rows,
            split.expected_fraud_rows,
        )
        for split in contract.splits
    ] == [
        (1, 432, 6_008_416, 4_857),
        (433, 576, 192_778, 1_516),
        (577, 743, 161_426, 1_840),
    ]


@pytest.mark.parametrize(
    "prohibited_source",
    [
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "old_balance_orig",
        "new_balance_orig",
        "old_balance_dest",
        "new_balance_dest",
        "isFraud",
        "is_fraud",
        "isFlaggedFraud",
        "is_flagged_fraud",
        "nameOrig",
        "nameDest",
        "name_orig",
        "name_dest",
        "step",
        "simulation_day",
        "simulation_hour_of_day",
    ],
)
def test_contract_rejects_prohibited_or_restricted_feature_dependencies(
    prohibited_source,
):
    document = _document()
    document["predictive_source_columns"].append(prohibited_source)
    document["features"].append(
        {
            "name": "innocent_sounding_feature",
            "source_columns": [prohibited_source],
            "transformation": "log1p",
            "dtype": "float32",
        }
    )

    with pytest.raises(
        FeatureContractError,
        match=rf"innocent_sounding_feature.*{prohibited_source}",
    ):
        parse_feature_contract(document)


def test_contract_rejects_non_chronological_or_overlapping_splits():
    document = copy.deepcopy(_document())
    document["splits"][1]["first_step"] = 432

    with pytest.raises(FeatureContractError, match="must begin immediately after"):
        parse_feature_contract(document)


def test_contract_requires_final_day_to_remain_in_test_partition():
    document = copy.deepcopy(_document())
    document["final_day"]["first_step"] = 500

    with pytest.raises(FeatureContractError, match="contained within the test split"):
        parse_feature_contract(document)


def test_contract_requires_expected_partition_counts_to_reconcile():
    document = copy.deepcopy(_document())
    document["splits"][0]["expected_rows"] -= 1

    with pytest.raises(FeatureContractError, match="reconcile to expected total rows"):
        parse_feature_contract(document)


def test_v2_internal_grouping_exception_does_not_allow_balance_leakage():
    document = json.loads(Path("config/features/paysim_v2.json").read_text(encoding="utf-8"))
    document["features"].append(
        {
            "name": "unsafe_history",
            "source_columns": ["name_dest", "old_balance_orig"],
            "transformation": "precomputed_history",
            "dtype": "float32",
        }
    )

    with pytest.raises(FeatureContractError, match="unsafe_history.*old_balance_orig"):
        parse_feature_contract(document)


def test_final_v2_contract_contains_only_approved_behavioural_features():
    contract = load_feature_contract("config/features/paysim_v2.json")

    assert len(contract.feature_names) == 20
    assert not {
        "relationship_prior_count_log1p",
        "is_new_origin_dest_relationship",
        "isolation_forest_anomaly_percentile",
        "anomaly_history_available",
    } & set(contract.feature_names)


def test_rejected_candidates_remain_evidence_only():
    evidence = json.loads(
        Path("config/experiments/paysim_rejected_feature_candidates_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert evidence["normal_pipeline_dependency"] is False
    assert evidence["test_partition_loaded"] is False
    assert evidence["relationship_features"]["decision"] == "excluded_from_selected_contract"
    assert evidence["isolation_forest"]["decision"] == "excluded_from_selected_contract"
