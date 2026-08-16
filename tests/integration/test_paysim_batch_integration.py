from __future__ import annotations

import csv
import logging
import os
from dataclasses import replace
from decimal import Decimal

import psycopg
import pytest

from src.data import paysim_ingestion
from src.data.paysim_ingestion import (
    IngestionRunError,
    RecordCounts,
    SourceSchemaError,
    create_staging_table,
    database_counts,
    ingest_paysim,
    parse_paysim_row,
    persist_chunk,
)
from src.data.postgres import load_postgres_settings
from src.data.transaction_identity import calculate_dataset_sha256


pytestmark = pytest.mark.integration


def _integration_tests_enabled() -> bool:
    return os.getenv("RUN_POSTGRES_TESTS") == "1"


@pytest.fixture(scope="module")
def connection():
    if not _integration_tests_enabled():
        pytest.skip("Set RUN_POSTGRES_TESTS=1 to run PostgreSQL integration tests")

    settings = load_postgres_settings()
    with psycopg.connect(**settings.connection_kwargs(), autocommit=True) as database_connection:
        yield database_connection


def _sample_row(**overrides):
    row = {
        "step": "1",
        "type": "PAYMENT",
        "amount": "0.10",
        "nameOrig": "C_BATCH_100",
        "oldbalanceOrg": "100.10",
        "newbalanceOrig": "100.00",
        "nameDest": "M_BATCH_200",
        "oldbalanceDest": "0.00",
        "newbalanceDest": "0.10",
        "isFraud": "0",
        "isFlaggedFraud": "0",
    }
    row.update(overrides)
    return row


def _write_csv(path, rows, columns=paysim_ingestion.PAYSIM_COLUMNS):
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def sample_csv(tmp_path, connection):
    source = tmp_path / "paysim-integration.csv"
    _write_csv(
        source,
        [
            _sample_row(),
            _sample_row(
                step="2",
                type="TRANSFER",
                amount="250.25",
                nameOrig="C_BATCH_101",
                nameDest="C_BATCH_201",
                isFraud="1",
            ),
            _sample_row(
                step="3",
                type="CASH_OUT",
                amount="999.99",
                nameOrig="C_BATCH_102",
                nameDest="C_BATCH_202",
                isFlaggedFraud="1",
            ),
        ],
    )
    dataset_id = calculate_dataset_sha256(source)
    connection.execute("DELETE FROM raw.transactions WHERE source_dataset_id = %s", (dataset_id,))
    yield source, dataset_id
    connection.execute("DELETE FROM raw.transactions WHERE source_dataset_id = %s", (dataset_id,))


def test_batch_ingestion_preserves_exact_values_reconciles_and_is_idempotent(
    connection, sample_csv
):
    source, dataset_id = sample_csv
    expected = RecordCounts(total=3, fraud=1, flagged_fraud=1)

    first = ingest_paysim(source, chunk_size=2, expected_totals=expected)

    assert first.processed == 3
    assert first.inserted == 3
    assert first.skipped == 0
    assert first.failed == 0
    assert first.reconciled is True
    stored = connection.execute(
        """
        SELECT amount, old_balance_orig, first_ingestion_source, source_row_number
        FROM raw.transactions
        WHERE source_dataset_id = %s
        ORDER BY source_row_number
        """,
        (dataset_id,),
    ).fetchall()
    assert stored == [
        (Decimal("0.10"), Decimal("100.10"), "batch", 1),
        (Decimal("250.25"), Decimal("100.10"), "batch", 2),
        (Decimal("999.99"), Decimal("100.10"), "batch", 3),
    ]

    second = ingest_paysim(source, chunk_size=2, expected_totals=expected)

    assert second.processed == 3
    assert second.inserted == 0
    assert second.skipped == 3
    assert second.failed == 0
    assert second.reconciled is True
    assert database_counts(connection, dataset_id) == expected


def test_incompatible_schema_stops_before_database_ingestion(connection, tmp_path):
    source = tmp_path / "missing-column.csv"
    columns = tuple(
        column for column in paysim_ingestion.PAYSIM_COLUMNS if column != "isFraud"
    )
    _write_csv(source, [_sample_row()], columns=columns)
    dataset_id = calculate_dataset_sha256(source)
    before = database_counts(connection, dataset_id).total

    with pytest.raises(SourceSchemaError, match="missing columns: isFraud"):
        ingest_paysim(source, chunk_size=2, expected_totals=None)

    after = database_counts(connection, dataset_id).total
    assert after == before


def test_database_batch_failure_rolls_back_all_rows(connection):
    dataset_id = "b" * 64
    first = parse_paysim_row(_sample_row(), 8_000_001, dataset_id, "failure-test.csv")
    second = parse_paysim_row(
        _sample_row(nameOrig="C_BATCH_FAILURE"),
        8_000_002,
        dataset_id,
        "failure-test.csv",
    )
    # This value bypasses source parsing intentionally to force PostgreSQL NUMERIC overflow.
    invalid_second = replace(second, amount=Decimal("1000000000000000000.00"))
    connection.execute("DELETE FROM raw.transactions WHERE source_dataset_id = %s", (dataset_id,))
    create_staging_table(connection)

    with pytest.raises(psycopg.Error):
        persist_chunk(connection, [first, invalid_second])

    assert database_counts(connection, dataset_id).total == 0


def test_failed_batch_is_counted_and_logged(connection, sample_csv, monkeypatch, caplog):
    source, _ = sample_csv

    def fail_persistence(_connection, _chunk):
        raise psycopg.OperationalError("simulated outage containing C_SENSITIVE_ROW")

    monkeypatch.setattr(paysim_ingestion, "persist_chunk", fail_persistence)
    with caplog.at_level(logging.ERROR), pytest.raises(IngestionRunError) as error:
        ingest_paysim(source, chunk_size=2, expected_totals=None)

    summary = error.value.summary
    assert summary.processed == 2
    assert summary.inserted == 0
    assert summary.skipped == 0
    assert summary.failed == 2
    assert "Database batch failed: chunk=1, rows=2, cause=OperationalError" in caplog.text
    assert "C_SENSITIVE_ROW" not in caplog.text
