"""Tests that promotion accepts only the approved, validation-only PH7-02 run."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.models.promote_scoring_model import ModelPromotionError, _validate_run


RUN_ID = "acd17ad55f8e4c2a85d0afedb3608a04"


def _run(contract_hash: str, *, test_loaded: str = "False"):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=RUN_ID),
        data=SimpleNamespace(
            params={
                "feature_contract_version": "paysim_features_v2",
                "feature_contract_sha256": contract_hash,
                "test_partition_loaded": test_loaded,
            },
            tags={"trial_name": "high_child_weight"},
        ),
    )


def test_promotion_accepts_approved_validation_only_identity():
    contract_hash = "a" * 64
    _validate_run(_run(contract_hash), {"mlflow_run_id": RUN_ID}, contract_hash)


def test_promotion_rejects_a_run_that_loaded_test_data():
    contract_hash = "a" * 64
    with pytest.raises(ModelPromotionError, match="test_partition_loaded"):
        _validate_run(
            _run(contract_hash, test_loaded="True"),
            {"mlflow_run_id": RUN_ID},
            contract_hash,
        )
