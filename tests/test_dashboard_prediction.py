"""Tests for the operational Streamlit-to-FastAPI scoring boundary."""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.api.contracts import PredictionResponse
from src.dashboard.prediction_client import (
    PredictionClient,
    PredictionRejectedError,
    PredictionServiceUnavailable,
)
from src.dashboard.pages.fraud_scoring import (
    _history_summary,
    _review_label,
    _technical_feature_table,
)
from src.dashboard.transaction_scoring import HistoricalTransactionScore
from src.features.contract import load_feature_contract
from src.features.point_in_time import PreparedTransactionFeatures, ReviewTransaction


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, status_code=200, payload=None) -> None:
        self.status_code = status_code
        self.payload = payload or {}

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, response=None, error=None) -> None:
        self.response = response
        self.error = error
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        if self.error:
            raise self.error
        return self.response


def _prepared(model_ready_payload) -> PreparedTransactionFeatures:
    contract = load_feature_contract("config/features/paysim_v2.json")
    frame = pd.DataFrame([model_ready_payload])
    for definition in contract.features:
        frame[definition.name] = frame[definition.name].astype(definition.dtype)
    return PreparedTransactionFeatures(
        transaction_id="bc7ae75a-3f04-5ad3-927b-741185385e87",
        source_dataset_id="a" * 64,
        source_row_number=2722,
        step=2,
        transaction_type="CASH_IN",
        amount=171_300.50,
        name_orig="C100",
        name_dest="C900",
        features=frame,
    )


def test_scoring_page_requests_only_a_transaction_id() -> None:
    app = AppTest.from_file(
        str(REPOSITORY_ROOT / "src/dashboard/pages/fraud_scoring.py"),
        default_timeout=30,
    ).run()

    assert not app.exception
    assert app.text_input[0].label == "Transaction ID"
    assert app.button(key="load_review_transactions")
    assert app.button(key="FormSubmitter:fraud_scoring_form-Analyse risk")
    assert len(app.number_input) == 0
    assert len(app.checkbox) == 0
    assert "Load demonstration example" not in " ".join(item.value for item in app.markdown)


def test_review_queue_labels_real_transaction_context_compactly() -> None:
    transaction = ReviewTransaction(
        transaction_id="bc7ae75a-3f04-5ad3-927b-741185385e87",
        source_row_number=2722,
        step=26,
        transaction_type="CASH_IN",
        amount=171_300.50,
        name_orig="C100",
        name_dest="C900",
    )

    assert _review_label(transaction) == (
        "Day 2, hour 2 · CASH_IN · 171,300.50 · bc7ae75a…5e87"
    )

    app = AppTest.from_file(
        str(REPOSITORY_ROOT / "src/dashboard/pages/fraud_scoring.py"),
        default_timeout=30,
    )
    app.session_state["review_transactions"] = (transaction,)
    app.run()

    assert not app.exception
    assert app.selectbox(key="review_transaction_selection").value == transaction
    assert any(button.label == "Analyse selected transaction" for button in app.button)
    assert len(app.number_input) == 0
    assert len(app.checkbox) == 0


def test_history_and_technical_details_are_read_only_presentations(
    model_ready_payload,
) -> None:
    prepared = _prepared(model_ready_payload)
    summary = _history_summary(prepared)
    table = _technical_feature_table(prepared)

    assert summary["previous_activity"] == "No earlier activity"
    assert list(table["Feature"]) == list(prepared.features.columns)
    assert len(table) == 20
    assert set(table) == {"Feature", "Value", "Data type"}


def test_scored_transaction_renders_context_probability_and_review_wording(
    model_ready_payload,
) -> None:
    result = HistoricalTransactionScore(
        prepared=_prepared(model_ready_payload),
        prediction=PredictionResponse(
            fraud_probability=0.82,
            risk_classification="high",
            model_version="paysim_fraud_model_v2_tuned_review",
            risk_policy_version="policy-v1",
            risk_policy_status="portfolio_review_only_not_production_approved",
        ),
    )
    app = AppTest.from_file(
        str(REPOSITORY_ROOT / "src/dashboard/pages/fraud_scoring.py"),
        default_timeout=30,
    )
    app.session_state["historical_transaction_score"] = result
    app.run()

    assert not app.exception
    assert any(metric.label == "Fraud probability" and metric.value == "82.00%" for metric in app.metric)
    assert any(item.value == "Transaction under review" for item in app.subheader)
    assert any(item.value == "Risk assessment" for item in app.subheader)
    assert any(item.label == "Technical details" for item in app.expander)
    assert any(item.label == "Model information" for item in app.expander)
    assert any(
        metric.label == "PR-AUC · ranking" and metric.value == "0.488"
        for metric in app.metric
    )
    assert any(
        metric.label == "PR-AUC · ranking" and metric.value == "0.540"
        for metric in app.metric
    )
    presentation = " ".join(item.value for item in app.markdown)
    assert "Validation evidence" in presentation
    assert "Held-out test evaluation" in presentation
    assert "does not by itself determine that fraud occurred" in " ".join(
        item.value for item in app.markdown
    )


def test_prediction_client_returns_typed_result(model_ready_payload) -> None:
    response = FakeResponse(
        payload={
            "fraud_probability": 0.82,
            "risk_classification": "high",
            "model_version": "reviewed-v2",
            "risk_policy_version": "policy-v1",
            "risk_policy_status": "portfolio_review_only_not_production_approved",
        }
    )
    transport = FakeClient(response=response)
    client = PredictionClient("http://127.0.0.1:8000", 5, transport)

    result = client.score(model_ready_payload)

    assert result.fraud_probability == 0.82
    assert result.risk_classification == "high"
    assert transport.requests[0][0].endswith("/v1/predictions")


def test_prediction_client_handles_validation_and_unavailable_service(
    model_ready_payload,
) -> None:
    rejected = PredictionClient(
        "http://127.0.0.1:8000", 5, FakeClient(response=FakeResponse(status_code=422))
    )
    with pytest.raises(PredictionRejectedError):
        rejected.score(model_ready_payload)

    unavailable = PredictionClient(
        "http://127.0.0.1:8000",
        5,
        FakeClient(error=httpx.ConnectError("private endpoint details")),
    )
    with pytest.raises(PredictionServiceUnavailable) as captured:
        unavailable.score(model_ready_payload)
    assert "private" not in str(captured.value)
