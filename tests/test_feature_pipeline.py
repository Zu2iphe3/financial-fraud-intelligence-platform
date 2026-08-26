"""Unit tests for leakage-safe preprocessing and chronological partitioning."""

from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from src.features.contract import load_feature_contract
from src.features.pipeline import (
    FeaturePipelineError,
    prepare_modelling_chunk,
    transform_features,
)


CONTRACT = load_feature_contract()


def _source_frame(**overrides):
    steps = [1, 432, 433, 576, 577, 720, 721, 743]
    values = {
        "transaction_id": [uuid4() for _ in steps],
        "source_dataset_id": ["a" * 64] * len(steps),
        "source_row_number": list(range(1, len(steps) + 1)),
        "step": steps,
        "simulation_day": [((step - 1) // 24) + 1 for step in steps],
        "transaction_type": [
            "CASH_IN",
            "CASH_OUT",
            "DEBIT",
            "PAYMENT",
            "TRANSFER",
            "PAYMENT",
            "TRANSFER",
            "CASH_OUT",
        ],
        "amount": [0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000],
        "is_fraud": [0, 0, 0, 0, 1, 0, 1, 1],
        # Unsafe source fields can exist in the trusted mart but are never selected as features.
        "old_balance_orig": [999] * len(steps),
        "is_flagged_fraud": [0] * len(steps),
        "name_orig": ["C_PRIVATE"] * len(steps),
    }
    values.update(overrides)
    return pd.DataFrame(values)


def test_preprocessing_produces_exact_contract_schema_and_values():
    source = _source_frame()

    features = transform_features(source, CONTRACT)

    assert tuple(features.columns) == CONTRACT.feature_names
    assert features["amount_log1p"].to_numpy() == pytest.approx(
        np.log1p(source["amount"]).astype("float32")
    )
    one_hot_columns = [column for column in features if column.startswith("transaction_type_")]
    assert features[one_hot_columns].sum(axis=1).tolist() == [1] * len(source)
    assert str(features["amount_log1p"].dtype) == "float32"
    assert all(str(features[column].dtype) == "int8" for column in one_hot_columns)


def test_target_metadata_and_features_are_strictly_separated():
    chunk = prepare_modelling_chunk(_source_frame(), CONTRACT)

    assert chunk.target.name == "is_fraud"
    assert "is_fraud" not in chunk.features
    assert "is_fraud" not in chunk.metadata
    assert not {
        "old_balance_orig",
        "is_flagged_fraud",
        "name_orig",
        "transaction_id",
        "step",
    }.intersection(chunk.features.columns)
    assert "transaction_id" in chunk.metadata
    assert str(chunk.metadata["transaction_id"].dtype) == "string"


def test_chronological_splits_and_final_day_cohorts_are_explicit():
    chunk = prepare_modelling_chunk(_source_frame(), CONTRACT)

    assert chunk.metadata["partition"].tolist() == [
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
        "test",
        "test",
    ]
    assert chunk.metadata["evaluation_cohort"].tolist()[-2:] == [
        "final_day_artifact",
        "final_day_artifact",
    ]
    assert set(chunk.metadata["evaluation_cohort"].tolist()[:-2]) == {"standard_period"}
    assert "evaluation_cohort" not in chunk.features


@pytest.mark.parametrize(
    ("column", "invalid_value", "message"),
    [
        ("amount", -1, "finite non-negative"),
        ("transaction_type", "CRYPTO", "Unsupported transaction type"),
        ("is_fraud", 2, "only 0 or 1"),
        ("step", 744, "outside configured splits"),
        ("simulation_day", 30, "does not match"),
    ],
)
def test_invalid_source_values_fail_before_dataset_generation(
    column, invalid_value, message
):
    source = _source_frame()
    source.loc[source.index[-1], column] = invalid_value

    with pytest.raises(FeaturePipelineError, match=message):
        prepare_modelling_chunk(source, CONTRACT)


def test_inference_transformation_needs_only_approved_predictive_inputs():
    inference_rows = pd.DataFrame(
        [{"amount": 250.0, "transaction_type": "TRANSFER"}]
    )

    features = transform_features(inference_rows, CONTRACT)

    assert tuple(features.columns) == CONTRACT.feature_names
    assert features.loc[0, "transaction_type_transfer"] == 1
