from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import psycopg
import pytest

from src.data.paysim_ingestion import parse_paysim_row
from src.data.postgres import load_postgres_settings
from src.streaming.consumer import KafkaPosition, persist_event
from src.streaming.event_contract import PaySimEvent, create_event_id


pytestmark = pytest.mark.integration
MIGRATION_SQL = (
    Path(__file__).parents[2] / "database" / "init" / "002_create_transaction_ingestions.sql"
)
DATASET_ID = "d" * 64


def _integration_tests_enabled() -> bool:
    return os.getenv("RUN_POSTGRES_TESTS") == "1"


@pytest.fixture(scope="module")
def connection():
    if not _integration_tests_enabled():
        pytest.skip("Set RUN_POSTGRES_TESTS=1 to run PostgreSQL integration tests")
    settings = load_postgres_settings()
    with psycopg.connect(**settings.connection_kwargs(), autocommit=True) as database_connection:
        yield database_connection


def _event(row_number: int = 9_100_001) -> PaySimEvent:
    transaction = parse_paysim_row(
        {
            "step": "5",
            "type": "TRANSFER",
            "amount": "125.25",
            "nameOrig": "C_STREAM_1",
            "oldbalanceOrg": "500.50",
            "newbalanceOrig": "375.25",
            "nameDest": "C_STREAM_2",
            "oldbalanceDest": "10.00",
            "newbalanceDest": "135.25",
            "isFraud": "0",
            "isFlaggedFraud": "0",
        },
        row_number,
        DATASET_ID,
        "stream.csv",
    )
    return PaySimEvent(
        create_event_id(transaction.transaction_id),
        "1.0",
        datetime(2026, 8, 17, tzinfo=timezone.utc),
        transaction,
    )


@pytest.fixture(autouse=True)
def clean_stream_records(connection):
    connection.execute(
        "DELETE FROM raw.transaction_ingestions WHERE transaction_id IN "
        "(SELECT transaction_id FROM raw.transactions WHERE source_dataset_id = %s)",
        (DATASET_ID,),
    )
    connection.execute("DELETE FROM raw.transactions WHERE source_dataset_id = %s", (DATASET_ID,))
    yield
    connection.execute(
        "DELETE FROM raw.transaction_ingestions WHERE transaction_id IN "
        "(SELECT transaction_id FROM raw.transactions WHERE source_dataset_id = %s)",
        (DATASET_ID,),
    )
    connection.execute("DELETE FROM raw.transactions WHERE source_dataset_id = %s", (DATASET_ID,))


def test_migration_is_repeatable_and_creates_audit_constraints(connection):
    connection.execute(MIGRATION_SQL.read_text(encoding="utf-8"))
    connection.execute(MIGRATION_SQL.read_text(encoding="utf-8"))

    constraints = {
        row[0]
        for row in connection.execute(
            """
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_schema = 'raw' AND table_name = 'transaction_ingestions'
            """
        ).fetchall()
    }
    assert "transaction_ingestions_pkey" in constraints
    assert "transaction_ingestions_topic_position_key" in constraints


def test_stream_persistence_is_atomic_and_idempotent(connection):
    event = _event()

    first = persist_event(connection, event, KafkaPosition("events", 0, 10))
    duplicate = persist_event(connection, event, KafkaPosition("events", 0, 11))

    assert first.transaction_inserted is True
    assert first.event_inserted is True
    assert duplicate.transaction_inserted is False
    assert duplicate.event_inserted is False
    transaction_count = connection.execute(
        "SELECT COUNT(*) FROM raw.transactions WHERE transaction_id = %s",
        (event.transaction.transaction_id,),
    ).fetchone()[0]
    ingestion_count = connection.execute(
        "SELECT COUNT(*) FROM raw.transaction_ingestions WHERE event_id = %s",
        (event.event_id,),
    ).fetchone()[0]
    assert transaction_count == ingestion_count == 1
    assert connection.execute(
        "SELECT first_ingestion_source FROM raw.transactions WHERE transaction_id = %s",
        (event.transaction.transaction_id,),
    ).fetchone() == ("kafka",)
    assert connection.execute(
        """
        SELECT event_version, kafka_topic, kafka_partition, kafka_offset
        FROM raw.transaction_ingestions WHERE event_id = %s
        """,
        (event.event_id,),
    ).fetchone() == ("1.0", "events", 0, 10)


def test_existing_identity_cannot_hide_changed_transaction_values(connection):
    event = _event()
    persist_event(connection, event, KafkaPosition("events", 0, 20))
    altered = replace(event, transaction=replace(event.transaction, amount=Decimal("999.99")))

    with pytest.raises(ValueError, match="different PaySim values"):
        persist_event(connection, altered, KafkaPosition("events", 0, 21))

    stored_amount = connection.execute(
        "SELECT amount FROM raw.transactions WHERE transaction_id = %s",
        (event.transaction.transaction_id,),
    ).fetchone()[0]
    assert stored_amount == Decimal("125.25")
