"""Shared small scoring fixtures independent of the versioned approved model."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.api.config import load_api_settings
from src.models.scoring_artifact import ScoringModelBundle, save_scoring_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FixedProbabilityEstimator:
    """Predict one controlled probability while recording actual inference calls."""

    def __init__(self, feature_count: int, probability: float = 0.8) -> None:
        self.n_features_in_ = feature_count
        self.probability = probability
        self.calls = 0
        self.raise_error = False

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.raise_error:
            raise RuntimeError(r"private failure at C:\Users\private\model.joblib")
        fraud = np.full(len(matrix), self.probability, dtype="float64")
        return np.column_stack((1 - fraud, fraud))


@pytest.fixture
def model_ready_payload() -> dict[str, float | int]:
    return json.loads(
        (REPOSITORY_ROOT / "tests/fixtures/model_ready_transaction.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture
def api_settings(tmp_path):
    settings = load_api_settings()
    return replace(settings, model_artifact_path=tmp_path / "model")


@pytest.fixture
def scoring_bundle(api_settings):
    estimator = FixedProbabilityEstimator(len(api_settings.feature_names))
    return ScoringModelBundle(
        model_version=api_settings.model_version,
        feature_contract_version=api_settings.feature_contract_version,
        feature_contract_sha256=api_settings.feature_contract_sha256,
        feature_names=api_settings.feature_names,
        estimator=estimator,
        selected_model=api_settings.selected_model,
        selected_run_id=api_settings.selected_run_id,
        package_versions={"python": "test"},
    )


@pytest.fixture
def saved_scoring_artifact(api_settings, scoring_bundle):
    save_scoring_bundle(scoring_bundle, api_settings.model_artifact_path)
    return api_settings.model_artifact_path
