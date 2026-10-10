"""Tests for versioned persistence of the model-ready scoring estimator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.scoring_artifact import (
    MANIFEST_FILENAME,
    ScoringArtifactError,
    load_scoring_bundle,
    save_scoring_bundle,
)


APPROVED_SCORING_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/models/paysim_fraud_model_v2_scoring"
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


def test_loader_rejects_model_bytes_that_do_not_match_manifest(
    api_settings, scoring_bundle, monkeypatch
):
    save_scoring_bundle(scoring_bundle, api_settings.model_artifact_path)
    model_path = api_settings.model_artifact_path / "model.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"tampered")

    # Integrity must fail before pickle-based joblib deserialization is attempted.
    called = False

    def unexpected_load(_path):
        nonlocal called
        called = True
        raise AssertionError("tampered model must not be deserialized")

    monkeypatch.setattr("src.models.scoring_artifact.joblib.load", unexpected_load)
    with pytest.raises(ScoringArtifactError, match="checksum"):
        load_scoring_bundle(api_settings.model_artifact_path)
    assert not called


def test_loader_rejects_manifest_without_model_checksum(api_settings, scoring_bundle):
    save_scoring_bundle(scoring_bundle, api_settings.model_artifact_path)
    path = api_settings.model_artifact_path / MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["model_sha256"]
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ScoringArtifactError, match="no valid model checksum"):
        load_scoring_bundle(api_settings.model_artifact_path)


def test_versioned_approved_scoring_bundle_is_loadable():
    bundle = load_scoring_bundle(APPROVED_SCORING_ARTIFACT)

    assert bundle.model_version == "paysim_fraud_model_v2_tuned_review"
    assert bundle.selected_run_id == "acd17ad55f8e4c2a85d0afedb3608a04"
    assert len(bundle.feature_names) == 20
