"""Versioned JSON contract shared by the PaySim producer and consumer."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from src.data.paysim_ingestion import PaySimTransaction, SourceRowError, parse_paysim_row


EVENT_VERSION = "1.0"
PAYSIM_EVENT_NAMESPACE = uuid.UUID("fa06be9c-e26d-540d-b6da-d3ab9954f808")


class EventValidationError(ValueError):
    """Raised when a Kafka message does not satisfy the supported event contract."""


@dataclass(frozen=True)
class PaySimEvent:
    """A validated event plus the exact PaySim transaction it represents."""

    event_id: uuid.UUID
    event_version: str
    published_at: datetime
    transaction: PaySimTransaction


def create_event_id(transaction_id: uuid.UUID) -> uuid.UUID:
    """Keep replay identity stable so re-sending one source row remains idempotent."""
    return uuid.uuid5(PAYSIM_EVENT_NAMESPACE, f"paysim-replay:{EVENT_VERSION}:{transaction_id}")


def build_event_payload(
    transaction: PaySimTransaction,
    published_at: datetime | None = None,
) -> bytes:
    """Serialize one PaySim row while visibly separating unsafe and label fields."""
    timestamp = published_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("published_at must include a timezone")

    payload = {
        "event_version": EVENT_VERSION,
        "event_id": str(create_event_id(transaction.transaction_id)),
        "transaction_id": str(transaction.transaction_id),
        "published_at": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "dataset_id": transaction.source_dataset_id,
            "row_number": transaction.source_row_number,
            "file_name": transaction.source_file_name,
        },
        "transaction": {
            "step": transaction.step,
            "type": transaction.transaction_type,
            "amount": format(transaction.amount, "f"),
            "nameOrig": transaction.name_orig,
            "nameDest": transaction.name_dest,
        },
        # The simulator's balance outputs demonstrate leakage and are never model inputs.
        "analysis_only_balances": {
            "oldbalanceOrg": format(transaction.old_balance_orig, "f"),
            "newbalanceOrig": format(transaction.new_balance_orig, "f"),
            "oldbalanceDest": format(transaction.old_balance_dest, "f"),
            "newbalanceDest": format(transaction.new_balance_dest, "f"),
        },
        # Labels remain available for evaluation, but are not part of inference input.
        "ground_truth": {
            "isFraud": transaction.is_fraud,
            "isFlaggedFraud": transaction.is_flagged_fraud,
        },
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise EventValidationError(f"{key} must be an object")
    return value


def _string(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise EventValidationError(f"{key} must be a non-empty string")
    return value


def _integer(parent: Mapping[str, Any], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EventValidationError(f"{key} must be an integer")
    return value


def _uuid(parent: Mapping[str, Any], key: str) -> uuid.UUID:
    try:
        return uuid.UUID(_string(parent, key))
    except ValueError as error:
        raise EventValidationError(f"{key} must be a UUID") from error


def parse_event_payload(payload: bytes | str) -> PaySimEvent:
    """Validate an untrusted Kafka value before any database write is attempted."""
    try:
        decoded = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        document = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise EventValidationError("message must be valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise EventValidationError("message root must be an object")

    event_version = _string(document, "event_version")
    if event_version != EVENT_VERSION:
        raise EventValidationError(
            f"unsupported event_version {event_version!r}; expected {EVENT_VERSION!r}"
        )

    event_id = _uuid(document, "event_id")
    supplied_transaction_id = _uuid(document, "transaction_id")
    source = _mapping(document, "source")
    transaction_data = _mapping(document, "transaction")
    balances = _mapping(document, "analysis_only_balances")
    labels = _mapping(document, "ground_truth")

    raw_row = {
        "step": _integer(transaction_data, "step"),
        "type": _string(transaction_data, "type"),
        "amount": _string(transaction_data, "amount"),
        "nameOrig": _string(transaction_data, "nameOrig"),
        "oldbalanceOrg": _string(balances, "oldbalanceOrg"),
        "newbalanceOrig": _string(balances, "newbalanceOrig"),
        "nameDest": _string(transaction_data, "nameDest"),
        "oldbalanceDest": _string(balances, "oldbalanceDest"),
        "newbalanceDest": _string(balances, "newbalanceDest"),
        "isFraud": _integer(labels, "isFraud"),
        "isFlaggedFraud": _integer(labels, "isFlaggedFraud"),
    }
    try:
        parsed_transaction = parse_paysim_row(
            raw_row,
            _integer(source, "row_number"),
            _string(source, "dataset_id"),
            _string(source, "file_name"),
        )
    except (SourceRowError, TypeError, ValueError) as error:
        raise EventValidationError(str(error)) from error

    if supplied_transaction_id != parsed_transaction.transaction_id:
        raise EventValidationError("transaction_id does not match dataset_id and row_number")
    if event_id != create_event_id(parsed_transaction.transaction_id):
        raise EventValidationError("event_id does not match the stable replay identity")

    try:
        published_at = datetime.fromisoformat(
            _string(document, "published_at").replace("Z", "+00:00")
        )
    except ValueError as error:
        raise EventValidationError("published_at must be an ISO-8601 timestamp") from error
    if published_at.tzinfo is None:
        raise EventValidationError("published_at must include a timezone")

    return PaySimEvent(
        event_id=event_id,
        event_version=event_version,
        published_at=published_at.astimezone(timezone.utc),
        transaction=parsed_transaction,
    )
