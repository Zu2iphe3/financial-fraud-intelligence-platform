"""Focused tests for safe single-transaction feature reconstruction."""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID

import numpy as np
import pandas as pd
import psycopg
import pytest

from src.api.contracts import PredictionResponse
from src.dashboard.transaction_scoring import HistoricalTransactionScoringService
from src.data.postgres import PostgresSettings
from src.features.contract import load_feature_contract
from src.features.point_in_time import (
    DESTINATION_HISTORY_QUERY,
    REVIEW_TRANSACTIONS_QUERY,
    FeatureHistoryUnavailableError,
    PointInTimeFeatureError,
    PointInTimeFeatureService,
    PreparedTransactionFeatures,
    ReviewTransaction,
    TransactionNotFoundError,
    reconstruct_feature_frame,
    reconstruct_from_connection,
)


TRANSACTION_ID = UUID("0f6d04f0-8d52-5e9d-b24f-6bb7dfd33f19")
DATASET_ID = "a" * 64
CONTRACT = load_feature_contract("config/features/paysim_v2.json")


def _selected(**overrides):
    row = {
        "transaction_id": TRANSACTION_ID,
        "source_dataset_id": DATASET_ID,
        "source_row_number": 100,
        "step": 200,
        "transaction_type": "TRANSFER",
        "amount": Decimal("999.50"),
        "name_orig": "C100",
        "name_dest": "C900",
    }
    row.update(overrides)
    return row


def _history(row_number: int, step: int, **overrides):
    row = {
        "source_dataset_id": DATASET_ID,
        "source_row_number": row_number,
        "step": step,
        "transaction_type": "PAYMENT",
        "amount": Decimal("100.00"),
        "name_orig": f"C{row_number}",
        "name_dest": "C900",
    }
    row.update(overrides)
    return row


def test_cold_start_preserves_exact_contract_order_and_dtypes() -> None:
    features = reconstruct_feature_frame(_selected(), [], CONTRACT)

    assert tuple(features) == CONTRACT.feature_names
    assert [str(features[name].dtype) for name in features] == [
        definition.dtype for definition in CONTRACT.features
    ]
    assert features.loc[0, "dest_history_available"] == 0
    assert features.loc[0, "dest_hours_since_previous_activity"] == 744
    historical = [name for name in features if name.startswith("dest_")]
    assert not features.loc[0, [name for name in historical if name != "dest_hours_since_previous_activity"]].any()
    assert features.loc[0, "amount_log_ratio_to_dest_mean_168h"] == 0


def test_history_windows_preserve_24h_and_168h_boundaries() -> None:
    rows = [
        _history(1, 31, amount=Decimal("5.00")),   # older than 168 hours
        _history(2, 32, amount=Decimal("10.00")),  # inclusive 168-hour boundary
        _history(3, 175, amount=Decimal("20.00")), # outside 24h, inside 168h
        _history(4, 176, amount=Decimal("30.00"), transaction_type="TRANSFER"),
    ]
    features = reconstruct_feature_frame(_selected(), rows, CONTRACT).iloc[0]

    assert features["dest_tx_count_log1p_24h"] == pytest.approx(np.log1p(1))
    assert features["dest_tx_count_log1p_168h"] == pytest.approx(np.log1p(3))
    assert features["dest_amount_sum_log1p_24h"] == pytest.approx(np.log1p(30))
    assert features["dest_amount_sum_log1p_168h"] == pytest.approx(np.log1p(60))
    assert features["dest_hours_since_previous_activity"] == 24
    assert features["dest_transfer_count_log1p_24h"] == pytest.approx(np.log1p(1))


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"source_dataset_id": "b" * 64}, "source dataset"),
        ({"name_dest": "C901"}, "destination"),
        ({"step": 200}, "same-step or future"),
        ({"step": 201}, "same-step or future"),
        ({"source_row_number": 100}, "selected transaction"),
    ],
)
def test_untrusted_history_cannot_cross_point_in_time_boundaries(changed, message) -> None:
    untrusted = _history(1, 199)
    untrusted.update(changed)
    with pytest.raises(PointInTimeFeatureError, match=message):
        reconstruct_feature_frame(_selected(), [untrusted], CONTRACT)


def test_prediction_payload_contains_only_json_safe_model_features() -> None:
    frame = reconstruct_feature_frame(_selected(), [_history(1, 199)], CONTRACT)
    prepared = PreparedTransactionFeatures(
        transaction_id=str(TRANSACTION_ID),
        source_dataset_id=DATASET_ID,
        source_row_number=100,
        step=200,
        transaction_type="TRANSFER",
        amount=999.50,
        name_orig="C100",
        name_dest="C900",
        features=frame,
    )
    payload = prepared.prediction_payload()

    assert tuple(payload) == CONTRACT.feature_names
    assert not set(CONTRACT.prohibited_predictive_sources) & set(payload)
    assert all(type(value) in {float, int} for value in payload.values())


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, selected, history):
        self.selected = selected
        self.history = history
        self.calls = []

    def execute(self, query, parameters):
        self.calls.append((query, parameters))
        return _Result([self.selected] if len(self.calls) == 1 and self.selected else self.history)


def test_database_lookup_filters_before_reconstruction() -> None:
    connection = _Connection(_selected(), [_history(1, 199)])
    prepared = reconstruct_from_connection(connection, TRANSACTION_ID, CONTRACT)

    assert prepared.transaction_id == str(TRANSACTION_ID)
    assert "step < %s" in DESTINATION_HISTORY_QUERY
    assert connection.calls[1][1] == (DATASET_ID, "C900", 200)


def test_review_queue_uses_real_rows_without_fraud_labels() -> None:
    row = _selected()

    class ReviewConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def execute(self, query, parameters):
            assert query == REVIEW_TRANSACTIONS_QUERY
            assert parameters == (20,)
            return _Result([row])

    def connect(**_):
        return ReviewConnection()

    settings = PostgresSettings("db", "user", "secret", "localhost", 5432)
    service = PointInTimeFeatureService(settings, contract=CONTRACT, connect=connect)
    transactions = service.list_review_transactions()

    assert transactions == (
        ReviewTransaction(
            transaction_id=str(TRANSACTION_ID),
            source_row_number=100,
            step=200,
            transaction_type="TRANSFER",
            amount=999.50,
            name_orig="C100",
            name_dest="C900",
        ),
    )
    assert transactions[0].short_transaction_id == "0f6d04f0…3f19"
    assert "fraud" not in REVIEW_TRANSACTIONS_QUERY.lower()


@pytest.mark.parametrize("limit", [0, 101, True])
def test_review_queue_rejects_unbounded_limits(limit) -> None:
    settings = PostgresSettings("db", "user", "secret", "localhost", 5432)
    service = PointInTimeFeatureService(settings, contract=CONTRACT)
    with pytest.raises(ValueError, match="between 1 and 100"):
        service.list_review_transactions(limit)


def test_missing_transaction_and_database_failure_are_explicit() -> None:
    with pytest.raises(TransactionNotFoundError, match="not found"):
        reconstruct_from_connection(_Connection(None, []), TRANSACTION_ID, CONTRACT)

    def failed_connect(**_):
        raise psycopg.OperationalError("database unavailable")

    settings = PostgresSettings("db", "user", "secret", "localhost", 5432)
    service = PointInTimeFeatureService(settings, contract=CONTRACT, connect=failed_connect)
    with pytest.raises(FeatureHistoryUnavailableError, match="unavailable"):
        service.reconstruct(TRANSACTION_ID)


def test_application_service_calls_prediction_only_after_reconstruction() -> None:
    frame = reconstruct_feature_frame(_selected(), [], CONTRACT)
    prepared = PreparedTransactionFeatures(
        transaction_id=str(TRANSACTION_ID),
        source_dataset_id=DATASET_ID,
        source_row_number=100,
        step=200,
        transaction_type="TRANSFER",
        amount=999.50,
        name_orig="C100",
        name_dest="C900",
        features=frame,
    )

    class Features:
        def reconstruct(self, transaction_id):
            assert transaction_id == TRANSACTION_ID
            return prepared

    class Predictions:
        payload = None

        def score(self, payload):
            self.payload = payload
            return PredictionResponse(
                fraud_probability=0.42,
                risk_classification="moderate",
                model_version="reviewed-model",
                risk_policy_version="reviewed-policy",
                risk_policy_status="portfolio_review_only_not_production_approved",
            )

    predictions = Predictions()
    result = HistoricalTransactionScoringService(Features(), predictions).score(TRANSACTION_ID)

    assert tuple(predictions.payload) == CONTRACT.feature_names
    assert result.prediction.fraud_probability == 0.42
    assert result.prepared.simulation_day == 9
    assert result.prepared.simulation_hour_of_day == 8
