from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.data.paysim_ingestion import parse_paysim_row
from src.streaming.event_contract import (
    EVENT_VERSION,
    EventValidationError,
    build_event_payload,
    create_event_id,
    parse_event_payload,
)


DATASET_ID = "a" * 64


def sample_transaction(row_number: int = 7):
    return parse_paysim_row(
        {
            "step": "12",
            "type": "TRANSFER",
            "amount": "100.10",
            "nameOrig": "C100",
            "oldbalanceOrg": "500.25",
            "newbalanceOrig": "400.15",
            "nameDest": "C200",
            "oldbalanceDest": "25.00",
            "newbalanceDest": "125.10",
            "isFraud": "1",
            "isFlaggedFraud": "0",
        },
        row_number,
        DATASET_ID,
        "paysim.csv",
    )


def test_event_round_trip_preserves_identity_and_exact_money():
    transaction = sample_transaction()
    published_at = datetime(2026, 8, 17, 10, 30, tzinfo=timezone.utc)

    payload = build_event_payload(transaction, published_at)
    event = parse_event_payload(payload)
    document = json.loads(payload)

    assert event.event_version == EVENT_VERSION
    assert event.event_id == create_event_id(transaction.transaction_id)
    assert event.published_at == published_at
    assert event.transaction == transaction
    assert event.transaction.amount == Decimal("100.10")
    assert document["transaction"]["amount"] == "100.10"


def test_contract_visibly_separates_labels_and_analysis_only_balances():
    document = json.loads(build_event_payload(sample_transaction()))

    assert set(document["transaction"]) == {"step", "type", "amount", "nameOrig", "nameDest"}
    assert set(document["analysis_only_balances"]) == {
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    }
    assert set(document["ground_truth"]) == {"isFraud", "isFlaggedFraud"}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.update(event_version="2.0"), "unsupported event_version"),
        (lambda item: item.update(transaction_id="not-a-uuid"), "transaction_id must be a UUID"),
        (
            lambda item: item["transaction"].update(amount="0.001"),
            "more than two decimal places",
        ),
        (lambda item: item["source"].update(row_number=0), "row_number"),
    ],
)
def test_invalid_contract_values_are_rejected(mutation, message):
    document = json.loads(build_event_payload(sample_transaction()))
    mutation(document)

    with pytest.raises(EventValidationError, match=message):
        parse_event_payload(json.dumps(document))


def test_stable_event_identity_changes_only_with_logical_transaction():
    first = sample_transaction(row_number=7)
    same = sample_transaction(row_number=7)
    different = sample_transaction(row_number=8)

    assert create_event_id(first.transaction_id) == create_event_id(same.transaction_id)
    assert create_event_id(first.transaction_id) != create_event_id(different.transaction_id)
