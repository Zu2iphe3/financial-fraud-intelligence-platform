"""FastAPI application that validates and scores model-ready fraud features."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.config import ApiSettings, load_api_settings
from src.api.contracts import PredictionRequest, PredictionResponse, ReadinessResponse
from src.api.model_service import InferenceError, ModelUnavailableError, PredictionService


LOGGER = logging.getLogger(__name__)


def create_app(
    settings: ApiSettings | None = None,
    service: PredictionService | None = None,
) -> FastAPI:
    """Build an injectable application while keeping production startup explicit."""
    resolved_settings = settings or load_api_settings()
    prediction_service = service or PredictionService(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        prediction_service.load()
        yield

    application = FastAPI(
        title=resolved_settings.service_name,
        version=resolved_settings.service_version,
        description=(
            "Scores a complete 20-feature PaySim v2 vector prepared by an upstream "
            "leakage-safe feature component. It does not calculate behavioural history."
        ),
        lifespan=lifespan,
    )
    application.state.prediction_service = prediction_service

    @application.middleware("http")
    async def request_identifier(request: Request, call_next):
        """Add correlation without logging private model inputs."""
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        fields = [
            {
                "field": ".".join(str(part) for part in item["loc"] if part != "body"),
                "message": item["msg"],
            }
            for item in error.errors()
        ]
        LOGGER.warning("prediction_request_rejected invalid_fields=%s", len(fields))
        return JSONResponse(status_code=422, content={"error": "invalid_request", "fields": fields})

    @application.get(
        "/health/ready",
        response_model=ReadinessResponse,
        summary="Report whether real model predictions are available",
    )
    async def readiness() -> JSONResponse:
        response = prediction_service.readiness()
        return JSONResponse(
            status_code=200 if response.prediction_available else 503,
            content=response.model_dump(),
        )

    @application.post(
        "/v1/predictions",
        response_model=PredictionResponse,
        summary="Score one upstream-prepared PaySim v2 feature vector",
    )
    async def predict(payload: PredictionRequest) -> PredictionResponse | JSONResponse:
        try:
            response = prediction_service.score(payload)
        except ModelUnavailableError:
            return JSONResponse(
                status_code=503,
                content={"error": "model_unavailable", "message": "Prediction is unavailable"},
            )
        except InferenceError:
            return JSONResponse(
                status_code=500,
                content={"error": "inference_failed", "message": "Prediction could not be completed"},
            )
        LOGGER.info(
            "prediction_completed model_version=%s risk_classification=%s",
            response.model_version,
            response.risk_classification,
        )
        return response

    return application


app = create_app()
