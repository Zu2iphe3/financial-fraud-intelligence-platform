"""Tests for deployable selected-model persistence and prediction parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.contract import load_feature_contract
from src.features.pipeline import transform_features
from src.models.artifact import (
    FraudModelBundle,
    ModelArtifactError,
    load_model_bundle,
    save_model_bundle,
)


def _fitted_bundle() -> tuple[FraudModelBundle, pd.DataFrame]:
    contract = load_feature_contract()
    raw = pd.DataFrame(
        {
            "amount": [10.0, 100.0, 1_000.0, 20_000.0, 500_000.0],
            "transaction_type": list(contract.transaction_types),
        }
    )
    features = transform_features(raw, contract)
    estimator = LogisticRegression(random_state=42).fit(
        features.to_numpy(dtype=np.float32), [0, 0, 0, 1, 1]
    )
    bundle = FraudModelBundle(
        model_version="test_model_v1",
        feature_contract=contract,
        feature_contract_sha256="b" * 64,
        estimator=estimator,
        threshold=0.4,
        selected_model="logistic_regression",
        selected_run_id="run-123",
        package_versions={"python": "test"},
    )
    return bundle, raw


def test_persisted_bundle_preserves_probabilities_and_schema(tmp_path):
    bundle, raw = _fitted_bundle()
    output = tmp_path / "model"
    before = bundle.predict_probabilities(raw)

    save_model_bundle(bundle, output)
    loaded = load_model_bundle(output)

    np.testing.assert_allclose(loaded.predict_probabilities(raw), before, rtol=0, atol=1e-12)
    assert loaded.input_schema["contract_version"] == "paysim_features_v1"
    assert loaded.predict(raw).shape == (5,)


def test_model_artifact_refuses_silent_overwrite(tmp_path):
    bundle, _ = _fitted_bundle()
    output = tmp_path / "model"
    save_model_bundle(bundle, output)

    with pytest.raises(ModelArtifactError, match="already exists"):
        save_model_bundle(bundle, output)
