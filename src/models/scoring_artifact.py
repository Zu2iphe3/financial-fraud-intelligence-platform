"""Persist a model-ready estimator without coupling scoring to feature preparation."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin


MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "model_manifest.json"
HASH_CHUNK_SIZE = 1024 * 1024


class ScoringArtifactError(ValueError):
    """Raised when a model-ready artifact cannot honour its reviewed contract."""


def calculate_file_sha256(path: str | Path) -> str:
    """Hash an artifact incrementally so integrity checks stay memory efficient."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ScoringModelBundle:
    """Versioned estimator that accepts only the ordered PH7 model-ready vector.

    Feature preparation deliberately remains outside this object. A future online
    feature service can replace today's upstream producer without changing scoring.
    """

    model_version: str
    feature_contract_version: str
    feature_contract_sha256: str
    feature_names: tuple[str, ...]
    estimator: ClassifierMixin
    selected_model: str
    selected_run_id: str
    package_versions: Mapping[str, str]

    def predict_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        """Validate model-vector identity before returning fraud probabilities."""
        if tuple(features.columns) != self.feature_names:
            raise ScoringArtifactError("Scoring feature names or order do not match the model")
        try:
            matrix = features.to_numpy(dtype=np.float32, copy=False)
        except (TypeError, ValueError) as error:
            raise ScoringArtifactError("Scoring features must be numeric") from error
        if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
            raise ScoringArtifactError("Scoring features must contain finite rows")
        probabilities = np.asarray(self.estimator.predict_proba(matrix), dtype=np.float64)
        if probabilities.shape != (len(matrix), 2):
            raise ScoringArtifactError("Estimator returned an unexpected probability shape")
        fraud = probabilities[:, 1]
        if not np.isfinite(fraud).all() or ((fraud < 0) | (fraud > 1)).any():
            raise ScoringArtifactError("Estimator returned an invalid fraud probability")
        return fraud

    @property
    def manifest_identity(self) -> tuple[object, ...]:
        """Return immutable fields that must agree with the readable manifest."""
        return (
            self.model_version,
            self.feature_contract_version,
            self.feature_contract_sha256,
            self.feature_names,
            self.selected_model,
            self.selected_run_id,
        )


def save_scoring_bundle(bundle: ScoringModelBundle, output_directory: str | Path) -> Path:
    """Write one reviewed scoring artifact without replacing an existing model."""
    destination = Path(output_directory)
    if destination.exists():
        raise ScoringArtifactError(
            f"Scoring artifact already exists: {destination}. Choose a new output."
        )
    destination.mkdir(parents=True)
    model_path = destination / MODEL_FILENAME
    joblib.dump(bundle, model_path)
    manifest = {
        "model_version": bundle.model_version,
        "feature_contract_version": bundle.feature_contract_version,
        "feature_contract_sha256": bundle.feature_contract_sha256,
        "feature_names": list(bundle.feature_names),
        "selected_model": bundle.selected_model,
        "selected_run_id": bundle.selected_run_id,
        "package_versions": dict(bundle.package_versions),
        "input_boundary": "model_ready_features_prepared_upstream",
        "model_file": MODEL_FILENAME,
        # Bind the readable model identity to the exact reviewed binary.
        "model_sha256": calculate_file_sha256(model_path),
    }
    (destination / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return model_path


def load_scoring_bundle(output_directory: str | Path) -> ScoringModelBundle:
    """Verify artifact integrity before loading and validating the estimator."""
    destination = Path(output_directory)
    try:
        manifest = json.loads((destination / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ScoringArtifactError(f"Cannot load scoring artifact: {type(error).__name__}") from error

    model_file = manifest.get("model_file")
    expected_sha256 = manifest.get("model_sha256")
    if model_file != MODEL_FILENAME:
        raise ScoringArtifactError("Scoring manifest identifies an unexpected model file")
    if not isinstance(expected_sha256, str) or re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ) is None:
        raise ScoringArtifactError("Scoring manifest has no valid model checksum")

    model_path = destination / model_file
    try:
        observed_sha256 = calculate_file_sha256(model_path)
    except OSError as error:
        raise ScoringArtifactError(f"Cannot load scoring artifact: {type(error).__name__}") from error
    if not hmac.compare_digest(observed_sha256, expected_sha256.lower()):
        raise ScoringArtifactError("Scoring model checksum does not match the manifest")

    try:
        # Never deserialize model bytes until their reviewed checksum is confirmed.
        bundle = joblib.load(model_path)
    except (OSError, TypeError, ValueError) as error:
        raise ScoringArtifactError(f"Cannot load scoring artifact: {type(error).__name__}") from error
    if not isinstance(bundle, ScoringModelBundle):
        raise ScoringArtifactError("Persisted object is not a ScoringModelBundle")
    observed = (
        manifest.get("model_version"),
        manifest.get("feature_contract_version"),
        manifest.get("feature_contract_sha256"),
        tuple(manifest.get("feature_names", [])),
        manifest.get("selected_model"),
        manifest.get("selected_run_id"),
    )
    if observed != bundle.manifest_identity:
        raise ScoringArtifactError("Scoring manifest does not match the persisted bundle")
    return bundle
