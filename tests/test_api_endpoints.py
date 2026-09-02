"""Endpoint tests mapped to PH8-01 readiness, validation, and error scenarios."""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.model_service import PredictionService


def _client(api_settings, loader) -> TestClient:
    service = PredictionService(api_settings, bundle_loader=loader)
    return TestClient(create_app(settings=api_settings, service=service))


def test_valid_vector_returns_probability_risk_and_model_version(
    api_settings, scoring_bundle, model_ready_payload
):
    with _client(api_settings, lambda _: scoring_bundle) as client:
        ready = client.get("/health/ready")
        response = client.post("/v1/predictions", json=model_ready_payload)

    assert ready.status_code == 200
    assert ready.json()["prediction_available"] is True
    assert response.status_code == 200
    assert response.json() == {
        "fraud_probability": 0.8,
        "risk_classification": "high",
        "model_version": api_settings.model_version,
        "risk_policy_version": api_settings.risk_policy.version,
        "risk_policy_status": "portfolio_review_only_not_production_approved",
    }
    assert response.headers["X-Request-ID"]


def test_missing_model_reports_unavailable_and_never_fabricates_prediction(
    api_settings, model_ready_payload
):
    def unavailable(_):
        raise FileNotFoundError("private local path")

    with _client(api_settings, unavailable) as client:
        ready = client.get("/health/ready")
        response = client.post("/v1/predictions", json=model_ready_payload)

    assert ready.status_code == 503
    assert ready.json()["prediction_available"] is False
    assert response.status_code == 503
    assert "probability" not in response.text
    assert "private" not in response.text


def test_incompatible_model_identity_reports_unavailable(
    api_settings, scoring_bundle, model_ready_payload
):
    incompatible = type(scoring_bundle)(
        **{**scoring_bundle.__dict__, "model_version": "unexpected-model"}
    )
    with _client(api_settings, lambda _: incompatible) as client:
        ready = client.get("/health/ready")
        prediction = client.post("/v1/predictions", json=model_ready_payload)

    assert ready.status_code == 503
    assert prediction.status_code == 503


def test_invalid_request_identifies_field_without_running_inference(
    api_settings, scoring_bundle, model_ready_payload
):
    model_ready_payload.pop("amount_log1p")
    estimator = scoring_bundle.estimator

    with _client(api_settings, lambda _: scoring_bundle) as client:
        response = client.post("/v1/predictions", json=model_ready_payload)

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    assert response.json()["fields"][0]["field"] == "amount_log1p"
    assert estimator.calls == 0


def test_prohibited_extra_field_is_rejected(api_settings, scoring_bundle, model_ready_payload):
    model_ready_payload["nameDest"] = "M123"
    with _client(api_settings, lambda _: scoring_bundle) as client:
        response = client.post("/v1/predictions", json=model_ready_payload)

    assert response.status_code == 422
    assert any(field["field"] == "nameDest" for field in response.json()["fields"])
    assert scoring_bundle.estimator.calls == 0


def test_unexpected_inference_failure_is_sanitized(
    api_settings, scoring_bundle, model_ready_payload, caplog
):
    scoring_bundle.estimator.raise_error = True
    with caplog.at_level(logging.ERROR), _client(
        api_settings, lambda _: scoring_bundle
    ) as client:
        response = client.post("/v1/predictions", json=model_ready_payload)

    assert response.status_code == 500
    assert response.json() == {
        "error": "inference_failed",
        "message": "Prediction could not be completed",
    }
    assert "C:\\Users\\private" not in response.text
    assert str(model_ready_payload["amount_log1p"]) not in caplog.text
    assert "C:\\Users\\private" not in caplog.text
