"""Application service joining point-in-time preparation to the scoring API."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from src.api.contracts import PredictionResponse
from src.dashboard.prediction_client import PredictionClient
from src.features.point_in_time import (
    PointInTimeFeatureService,
    PreparedTransactionFeatures,
    ReviewTransaction,
)


@dataclass(frozen=True)
class HistoricalTransactionScore:
    """Trace one real prediction back to its reconstructed transaction vector."""

    prepared: PreparedTransactionFeatures
    prediction: PredictionResponse


class HistoricalTransactionScoringService:
    """Score stored transactions without exposing raw identifiers to the model API."""

    def __init__(
        self,
        feature_service: PointInTimeFeatureService,
        prediction_client: PredictionClient,
    ) -> None:
        self.feature_service = feature_service
        self.prediction_client = prediction_client

    def prepare(self, transaction_id: str | UUID) -> PreparedTransactionFeatures:
        """Load the transaction and reconstruct its reviewed point-in-time vector."""
        return self.feature_service.reconstruct(transaction_id)

    def list_review_transactions(self, limit: int = 20) -> tuple[ReviewTransaction, ...]:
        """Expose a compact real-data investigation queue without a second score path."""
        return self.feature_service.list_review_transactions(limit)

    def score_prepared(
        self, prepared: PreparedTransactionFeatures
    ) -> HistoricalTransactionScore:
        """Submit only a successfully reconstructed vector to FastAPI."""
        prediction = self.prediction_client.score(prepared.prediction_payload())
        return HistoricalTransactionScore(prepared=prepared, prediction=prediction)

    def score(self, transaction_id: str | UUID) -> HistoricalTransactionScore:
        """Reconstruct first; call FastAPI only after valid features exist."""
        return self.score_prepared(self.prepare(transaction_id))
