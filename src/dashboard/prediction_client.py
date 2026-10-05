"""FastAPI client and safe presentation mapping for fraud-risk scoring."""

from __future__ import annotations

from typing import Any, Mapping

import httpx

from src.api.contracts import PredictionRequest, PredictionResponse


class PredictionServiceUnavailable(RuntimeError):
    """Raised when Streamlit cannot reach a ready FastAPI scoring service."""


class PredictionRejectedError(ValueError):
    """Raised when the API rejects a reconstructed model-ready request."""


class PredictionClient:
    """Call FastAPI with bounded timeouts and controlled dashboard errors."""

    def __init__(self, base_url: str, timeout_seconds: float, client: Any | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client = client or httpx.Client(timeout=timeout_seconds)

    def score(self, payload: Mapping[str, Any]) -> PredictionResponse:
        validated = PredictionRequest.model_validate(payload)
        try:
            response = self.client.post(
                f"{self.base_url}/v1/predictions",
                json=validated.model_dump(), timeout=self.timeout_seconds,
            )
        except (httpx.HTTPError, OSError) as error:
            raise PredictionServiceUnavailable("Prediction service is unavailable") from error

        if response.status_code == 422:
            raise PredictionRejectedError("The scoring service rejected the supplied features")
        if response.status_code != 200:
            raise PredictionServiceUnavailable("Prediction service is unavailable")
        try:
            return PredictionResponse.model_validate(response.json())
        except ValueError as error:
            raise PredictionServiceUnavailable(
                "Prediction service returned an invalid response"
            ) from error
