"""Persist a model-ready estimator without coupling scoring to feature preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin


MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "model_manifest.json"


class ScoringArtifactError(ValueError):
    """Raised when a model-ready artifact cannot honour its reviewed contract."""


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
    }
    (destination / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return model_path


def load_scoring_bundle(output_directory: str | Path) -> ScoringModelBundle:
    """Load an estimator and reject a missing, corrupt, or mismatched manifest."""
    destination = Path(output_directory)
    try:
        manifest = json.loads((destination / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        bundle = joblib.load(destination / manifest["model_file"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
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
