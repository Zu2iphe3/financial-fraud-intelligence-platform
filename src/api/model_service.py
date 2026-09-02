"""Load one scoring bundle and keep inference independent from feature preparation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from src.api.config import ApiSettings
from src.api.contracts import PredictionRequest, PredictionResponse, ReadinessResponse
from src.models.scoring_artifact import (
    ScoringArtifactError,
    ScoringModelBundle,
    load_scoring_bundle,
)


LOGGER = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when startup did not establish a compatible scoring model."""


class InferenceError(RuntimeError):
    """Raised when a validated vector cannot be scored safely."""


BundleLoader = Callable[[str | Path], ScoringModelBundle]


class PredictionService:
    """Own model readiness and score validated, upstream-prepared feature vectors."""

    def __init__(
        self,
        settings: ApiSettings,
        bundle_loader: BundleLoader = load_scoring_bundle,
    ) -> None:
        self.settings = settings
        self._bundle_loader = bundle_loader
        self._bundle: ScoringModelBundle | None = None
        self._load_error: str | None = None

    def load(self) -> None:
        """Load once at startup and retain only a safe error category on failure."""
        try:
            bundle = self._bundle_loader(self.settings.model_artifact_path)
            expected = (
                self.settings.model_version,
                self.settings.feature_contract_version,
                self.settings.feature_contract_sha256,
                self.settings.feature_names,
                self.settings.selected_model,
                self.settings.selected_run_id,
            )
            if bundle.manifest_identity != expected:
                raise ScoringArtifactError("Configured model identity does not match artifact")
        except Exception as error:
            self._bundle = None
            self._load_error = type(error).__name__
            LOGGER.error("prediction_model_load_failed error_type=%s", self._load_error)
            return
        self._bundle = bundle
        self._load_error = None
        LOGGER.info("prediction_model_ready model_version=%s", bundle.model_version)

    def readiness(self) -> ReadinessResponse:
        """Report availability without exposing artifact paths or internal errors."""
        ready = self._bundle is not None
        return ReadinessResponse(
            status="ready" if ready else "unavailable",
            prediction_available=ready,
            model_version=self._bundle.model_version if self._bundle else None,
            feature_contract_version=self.settings.feature_contract_version,
        )

    def score(self, request: PredictionRequest) -> PredictionResponse:
        """Score one complete vector; validation and history building stay upstream."""
        if self._bundle is None:
            raise ModelUnavailableError("Prediction model is unavailable")
        try:
            frame = request.to_frame(self.settings.feature_names)
            probability = float(self._bundle.predict_probabilities(frame)[0])
        except Exception as error:
            # Do not attach exception text or traceback: estimators can include local
            # paths or values in failures, while the error type is enough to diagnose.
            LOGGER.error("prediction_inference_failed error_type=%s", type(error).__name__)
            raise InferenceError("Prediction could not be completed") from error
        return PredictionResponse(
            fraud_probability=probability,
            risk_classification=self.settings.risk_policy.classify(probability),
            model_version=self._bundle.model_version,
            risk_policy_version=self.settings.risk_policy.version,
            risk_policy_status=self.settings.risk_policy.status,
        )
