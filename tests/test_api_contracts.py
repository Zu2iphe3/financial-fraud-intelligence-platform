"""Tests for the exact public, leakage-safe model-vector schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.app import create_app
from src.api.contracts import PredictionRequest


PROHIBITED_FIELDS = {
    "nameOrig",
    "nameDest",
    "name_orig",
    "name_dest",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
    "step",
    "simulation_day",
}


def test_request_matches_exact_feature_order(api_settings, model_ready_payload):
    request = PredictionRequest.model_validate(model_ready_payload)
    assert tuple(request.to_frame(api_settings.feature_names).columns) == api_settings.feature_names


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("amount_log1p", float("nan")),
        ("dest_hours_since_previous_activity", 745),
        ("dest_history_available", 2),
        ("dest_tx_count_log1p_24h", -1),
        ("dest_amount_sum_log1p_168h", -1),
    ],
)
def test_request_rejects_invalid_numeric_domains(model_ready_payload, field, value):
    model_ready_payload[field] = value
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(model_ready_payload)


def test_request_requires_exactly_one_transaction_type(model_ready_payload):
    model_ready_payload["transaction_type_cash_out"] = 1
    with pytest.raises(ValidationError, match="exactly one"):
        PredictionRequest.model_validate(model_ready_payload)


def test_openapi_exposes_only_approved_model_features(api_settings):
    schema = create_app(settings=api_settings).openapi()
    properties = schema["components"]["schemas"]["PredictionRequest"]["properties"]

    assert tuple(properties) == api_settings.feature_names
    assert not (set(properties) & PROHIBITED_FIELDS)
