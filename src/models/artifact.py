"""Persist the selected raw-input preprocessing, estimator, and threshold together."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from src.features.contract import FeatureContract
from src.features.pipeline import transform_features


MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "model_manifest.json"


class ModelArtifactError(ValueError):
    """Raised when a persisted model cannot honour its versioned input contract."""


@dataclass(frozen=True)
class FraudModelBundle:
    """Deployable raw-transaction transformer, classifier, and decision policy."""

    model_version: str
    feature_contract: FeatureContract
    feature_contract_sha256: str
    estimator: ClassifierMixin
    threshold: float
    selected_model: str
    selected_run_id: str
    package_versions: Mapping[str, str]

    @property
    def input_schema(self) -> dict[str, Any]:
        """Describe the exact raw fields that a later FastAPI endpoint must accept."""
        return {
            "contract_version": self.feature_contract.contract_version,
            "fields": [
                {"name": "amount", "type": "non-negative finite number"},
                {
                    "name": "transaction_type",
                    "type": "string",
                    "allowed_values": list(self.feature_contract.transaction_types),
                },
            ],
            "predictive_features": list(self.feature_contract.feature_names),
        }

    def predict_probabilities(self, raw_transactions: pd.DataFrame) -> np.ndarray:
        """Apply the reviewed PH7-01 transform before calculating fraud probability."""
        features = transform_features(raw_transactions, self.feature_contract)
        probabilities = self.estimator.predict_proba(
            features.to_numpy(dtype=np.float32, copy=False)
        )
        return np.asarray(probabilities[:, 1], dtype=np.float64)

    def predict(self, raw_transactions: pd.DataFrame) -> np.ndarray:
        """Apply the persisted validation-selected operating threshold."""
        return (self.predict_probabilities(raw_transactions) >= self.threshold).astype(np.int8)


def save_model_bundle(bundle: FraudModelBundle, output_directory: str | Path) -> Path:
    """Write one versioned artifact without silently replacing a reviewed model."""
    destination = Path(output_directory)
    if destination.exists():
        raise ModelArtifactError(
            f"Model artifact already exists: {destination}. Choose a new output or remove it deliberately."
        )
    destination.mkdir(parents=True)
    model_path = destination / MODEL_FILENAME
    joblib.dump(bundle, model_path)
    manifest = {
        "model_version": bundle.model_version,
        "selected_model": bundle.selected_model,
        "selected_run_id": bundle.selected_run_id,
        "threshold": bundle.threshold,
        "feature_contract_version": bundle.feature_contract.contract_version,
        "feature_contract_sha256": bundle.feature_contract_sha256,
        "input_schema": bundle.input_schema,
        "package_versions": dict(bundle.package_versions),
        "model_file": MODEL_FILENAME,
    }
    (destination / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return model_path


def load_model_bundle(output_directory: str | Path) -> FraudModelBundle:
    """Load a selected artifact without retraining and verify its manifest identity."""
    destination = Path(output_directory)
    try:
        manifest = json.loads((destination / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        bundle = joblib.load(destination / manifest["model_file"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ModelArtifactError(f"Cannot load model artifact {destination}: {error}") from error
    if not isinstance(bundle, FraudModelBundle):
        raise ModelArtifactError("Persisted object is not a FraudModelBundle")
    expected = (
        bundle.model_version,
        bundle.selected_model,
        bundle.selected_run_id,
        bundle.threshold,
        bundle.feature_contract.contract_version,
        bundle.feature_contract_sha256,
    )
    observed = (
        manifest.get("model_version"),
        manifest.get("selected_model"),
        manifest.get("selected_run_id"),
        manifest.get("threshold"),
        manifest.get("feature_contract_version"),
        manifest.get("feature_contract_sha256"),
    )
    if observed != expected:
        raise ModelArtifactError("Model manifest does not match the persisted bundle")
    return bundle
