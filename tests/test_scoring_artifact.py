"""Tests for versioned persistence of the model-ready scoring estimator."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.models.scoring_artifact import (
    MANIFEST_FILENAME,
    ScoringArtifactError,
    load_scoring_bundle,
    save_scoring_bundle,
)


def test_scoring_bundle_round_trip_preserves_identity_and_probability(
    api_settings, scoring_bundle, model_ready_payload
):
    frame = pd.DataFrame([model_ready_payload], columns=api_settings.feature_names)
    expected = scoring_bundle.predict_probabilities(frame)
    save_scoring_bundle(scoring_bundle, api_settings.model_artifact_path)

    loaded = load_scoring_bundle(api_settings.model_artifact_path)

    assert loaded.manifest_identity == scoring_bundle.manifest_identity
    np.testing.assert_allclose(loaded.predict_probabilities(frame), expected)


def test_scoring_bundle_rejects_contract_order_drift(api_settings, scoring_bundle):
    frame = pd.DataFrame([[0] * 20], columns=reversed(api_settings.feature_names))
    with pytest.raises(ScoringArtifactError, match="names or order"):
        scoring_bundle.predict_probabilities(frame)


def test_loader_rejects_tampered_manifest(api_settings, scoring_bundle):
    save_scoring_bundle(scoring_bundle, api_settings.model_artifact_path)
    path = api_settings.model_artifact_path / MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["selected_run_id"] = "different-run"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ScoringArtifactError, match="does not match"):
        load_scoring_bundle(api_settings.model_artifact_path)
