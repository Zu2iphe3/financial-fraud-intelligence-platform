from __future__ import annotations

import os
import uuid
from decimal import Decimal
from pathlib import Path

import psycopg
import pytest

from src.data.postgres import load_postgres_settings
from src.data.transaction_identity import create_transaction_id


pytestmark = pytest.mark.integration
SCHEMA_SQL = Path(__file__).parents[2] / "database" / "init" / "001_create_raw_transactions.sql"
EXPECTED_SOURCE_COLUMNS = {
    "step",
    "transaction_type",
    "amount",
    "name_orig",
    "old_balance_orig",
    "new_balance_orig",
    "name_dest",
    "old_balance_dest",
    "new_balance_dest",
    "is_fraud",
    "is_flagged_fraud",
}
EXPECTED_SOURCE_TYPES = {
    "step": "integer",
    "transaction_type": "character varying",
    "amount": "numeric",
    "name_orig": "text",
    "old_balance_orig": "numeric",
    "new_balance_orig": "numeric",
    "name_dest": "text",
    "old_balance_dest": "numeric",
    "new_balance_dest": "numeric",
    "is_fraud": "smallint",
    "is_flagged_fraud": "smallint",
}
EXPECTED_METADATA_COLUMNS = {
    "record_id",
    "transaction_id",
    "source_dataset_id",
    "source_row_number",
    "first_ingestion_source",
    "source_file_name",
    "ingested_at",
}


def _integration_tests_enabled() -> bool:
    return os.getenv("RUN_POSTGRES_TESTS") == "1"


@pytest.fixture(scope="module")
def connection():
    if not _integration_tests_enabled():
        pytest.skip("Set RUN_POSTGRES_TESTS=1 to run PostgreSQL integration tests")

    settings = load_postgres_settings()
    with psycopg.connect(**settings.connection_kwargs(), autocommit=True) as database_connection:
        yield database_connection


def _transaction_values(row_number: int = 9000001) -> tuple:
    dataset_id = "f" * 64
    return (
        create_transaction_id(dataset_id, row_number),
        dataset_id,
        row_number,
        1,
        "TRANSFER",
        Decimal("100.25"),
        "C100",
        Decimal("500.50"),
        Decimal("400.25"),
        "C200",
        Decimal("10.00"),
        Decimal("110.25"),
        0,
        0,
        "batch",
        "paysim.csv",
    )


INSERT_TRANSACTION = """
    INSERT INTO raw.transactions (
        transaction_id, source_dataset_id, source_row_number,
        step, transaction_type, amount, name_orig,
        old_balance_orig, new_balance_orig, name_dest,
        old_balance_dest, new_balance_dest, is_fraud, is_flagged_fraud,
        first_ingestion_source, source_file_name
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def test_database_accepts_connections(connection):
    assert connection.execute("SELECT 1").fetchone() == (1,)


def test_schema_maps_all_source_fields_and_separate_metadata(connection):
    rows = connection.execute(
        """
        SELECT column_name, data_type, numeric_precision, numeric_scale
        FROM information_schema.columns
        WHERE table_schema = 'raw' AND table_name = 'transactions'
        """
    ).fetchall()
    columns = {row[0] for row in rows}
    column_types = {row[0]: row[1] for row in rows}
    decimal_shapes = {row[0]: (row[2], row[3]) for row in rows if row[1] == "numeric"}

    assert EXPECTED_SOURCE_COLUMNS <= columns
    assert EXPECTED_METADATA_COLUMNS <= columns
    assert EXPECTED_SOURCE_COLUMNS.isdisjoint(EXPECTED_METADATA_COLUMNS)
    assert {name: column_types[name] for name in EXPECTED_SOURCE_TYPES} == EXPECTED_SOURCE_TYPES
    assert set(decimal_shapes) == {
        "amount",
        "old_balance_orig",
        "new_balance_orig",
        "old_balance_dest",
        "new_balance_dest",
    }
    assert set(decimal_shapes.values()) == {(20, 2)}
    assert len(columns) == 18


def test_valid_transaction_round_trip_preserves_decimal_values(connection):
    values = _transaction_values()
    try:
        connection.execute(INSERT_TRANSACTION, values)
        stored = connection.execute(
            """
            SELECT amount, old_balance_orig, new_balance_dest
            FROM raw.transactions WHERE transaction_id = %s
            """,
            (values[0],),
        ).fetchone()
        assert stored == (Decimal("100.25"), Decimal("500.50"), Decimal("110.25"))
    finally:
        connection.execute("DELETE FROM raw.transactions WHERE transaction_id = %s", (values[0],))


@pytest.mark.parametrize("label_index", [12, 13])
@pytest.mark.parametrize("invalid_value", [-1, 2])
def test_invalid_binary_labels_are_rejected(connection, label_index, invalid_value):
    values = list(_transaction_values(row_number=9000010 + invalid_value))
    values[label_index] = invalid_value

    with pytest.raises(psycopg.errors.CheckViolation):
        connection.execute(INSERT_TRANSACTION, values)


def test_logical_transaction_identity_is_unique(connection):
    values = _transaction_values(row_number=9000020)
    try:
        connection.execute(INSERT_TRANSACTION, values)
        duplicate_row = list(values)
        duplicate_row[2] = 9000021

        with pytest.raises(psycopg.errors.UniqueViolation):
            connection.execute(INSERT_TRANSACTION, duplicate_row)
    finally:
        connection.execute("DELETE FROM raw.transactions WHERE transaction_id = %s", (values[0],))


def test_dataset_row_identity_is_unique(connection):
    values = _transaction_values(row_number=9000030)
    try:
        connection.execute(INSERT_TRANSACTION, values)
        duplicate_identity = list(values)
        duplicate_identity[0] = uuid.uuid4()

        with pytest.raises(psycopg.errors.UniqueViolation):
            connection.execute(INSERT_TRANSACTION, duplicate_identity)
    finally:
        connection.execute("DELETE FROM raw.transactions WHERE transaction_id = %s", (values[0],))


def test_initialization_is_repeatable_and_preserves_existing_data(connection):
    values = _transaction_values(row_number=9000040)
    try:
        connection.execute(INSERT_TRANSACTION, values)
        connection.execute(SCHEMA_SQL.read_text(encoding="utf-8"))

        count = connection.execute(
            "SELECT COUNT(*) FROM raw.transactions WHERE transaction_id = %s",
            (values[0],),
        ).fetchone()[0]
        assert count == 1
    finally:
        connection.execute("DELETE FROM raw.transactions WHERE transaction_id = %s", (values[0],))


def test_expected_indexes_exist(connection):
    rows = connection.execute(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'raw' AND tablename = 'transactions'
        """
    ).fetchall()
    indexes = {row[0] for row in rows}

    assert {
        "transactions_pkey",
        "transactions_transaction_id_key",
        "transactions_dataset_row_key",
        "transactions_step_idx",
        "transactions_type_step_idx",
        "transactions_fraud_idx",
    } <= indexes
