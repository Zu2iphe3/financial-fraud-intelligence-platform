"""Exercise FastAPI through a real fitted estimator and persisted scoring bundle."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.api.app import create_app
from src.api.model_service import PredictionService
from src.models.scoring_artifact import ScoringModelBundle, save_scoring_bundle


def test_persisted_model_loads_and_scores_through_http(
    api_settings, model_ready_payload
):
    # A tiny fitted estimator proves serialization and HTTP inference without loading
    # the 6.3-million-row dataset or depending on ignored local MLflow state.
    random = np.random.default_rng(42)
    matrix = random.normal(size=(12, len(api_settings.feature_names))).astype("float32")
    estimator = LogisticRegression(random_state=42).fit(
        matrix, np.array([0, 1] * 6, dtype="int8")
    )
    bundle = ScoringModelBundle(
        model_version=api_settings.model_version,
        feature_contract_version=api_settings.feature_contract_version,
        feature_contract_sha256=api_settings.feature_contract_sha256,
        feature_names=api_settings.feature_names,
        estimator=estimator,
        selected_model=api_settings.selected_model,
        selected_run_id=api_settings.selected_run_id,
        package_versions={"python": "test"},
    )
    save_scoring_bundle(bundle, api_settings.model_artifact_path)
    service = PredictionService(api_settings)

    with TestClient(create_app(settings=api_settings, service=service)) as client:
        ready = client.get("/health/ready")
        prediction = client.post("/v1/predictions", json=model_ready_payload)

    assert ready.status_code == 200
    assert prediction.status_code == 200
    assert 0 <= prediction.json()["fraud_probability"] <= 1
    assert prediction.json()["model_version"] == api_settings.model_version
