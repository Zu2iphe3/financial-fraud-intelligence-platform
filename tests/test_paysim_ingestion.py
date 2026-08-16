from __future__ import annotations

import csv
from decimal import Decimal

import pytest

from src.data import paysim_ingestion
from src.data.paysim_ingestion import (
    IngestionRunError,
    IngestionSummary,
    PaySimIngestionError,
    RecordCounts,
    SourceRowError,
    SourceSchemaError,
    configured_chunk_size,
    discover_paysim_csv,
    iter_paysim_chunks,
    parse_paysim_row,
    reconcile_ingestion,
    validate_csv_header,
    validate_source_columns,
)


DATASET_ID = "a" * 64


def sample_row(**overrides):
    row = {
        "step": "1",
        "type": "PAYMENT",
        "amount": "0.10",
        "nameOrig": "C100",
        "oldbalanceOrg": "100.10",
        "newbalanceOrig": "100.00",
        "nameDest": "M200",
        "oldbalanceDest": "0.00",
        "newbalanceDest": "0.10",
        "isFraud": "0",
        "isFlaggedFraud": "0",
    }
    row.update(overrides)
    return row


def write_csv(path, rows, columns=paysim_ingestion.PAYSIM_COLUMNS):
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_discover_paysim_csv_finds_only_repository_raw_csv(tmp_path, monkeypatch):
    raw_directory = tmp_path / "data" / "raw"
    raw_directory.mkdir(parents=True)
    expected = raw_directory / "paysim.csv"
    write_csv(expected, [sample_row()])
    monkeypatch.setattr(paysim_ingestion, "RAW_DATA_DIRECTORY", raw_directory)

    assert discover_paysim_csv() == expected.resolve()


def test_discover_paysim_csv_rejects_missing_and_ambiguous_sources(tmp_path, monkeypatch):
    raw_directory = tmp_path / "data" / "raw"
    raw_directory.mkdir(parents=True)
    monkeypatch.setattr(paysim_ingestion, "RAW_DATA_DIRECTORY", raw_directory)

    with pytest.raises(PaySimIngestionError, match="No PaySim CSV"):
        discover_paysim_csv()

    write_csv(raw_directory / "first.csv", [sample_row()])
    write_csv(raw_directory / "second.csv", [sample_row()])
    with pytest.raises(PaySimIngestionError, match="Multiple CSV files.*first.csv.*second.csv"):
        discover_paysim_csv()


def test_validate_source_columns_accepts_only_documented_contract():
    validate_source_columns(paysim_ingestion.PAYSIM_COLUMNS)

    missing = tuple(
        column for column in paysim_ingestion.PAYSIM_COLUMNS if column != "isFraud"
    )
    with pytest.raises(SourceSchemaError, match="missing columns: isFraud"):
        validate_source_columns(missing)


def test_validate_csv_header_fails_before_reading_rows(tmp_path):
    source = tmp_path / "wrong.csv"
    write_csv(source, [sample_row()], columns=paysim_ingestion.PAYSIM_COLUMNS[:-1])

    with pytest.raises(SourceSchemaError, match="isFlaggedFraud"):
        validate_csv_header(source)


def test_iter_paysim_chunks_preserves_rows_and_exact_decimals(tmp_path):
    source = tmp_path / "paysim.csv"
    write_csv(
        source,
        [
            sample_row(amount="0.10"),
            sample_row(amount="123456789.01", nameOrig="C101"),
            sample_row(amount="0.00", nameOrig="C102"),
        ],
    )

    chunks = list(iter_paysim_chunks(source, DATASET_ID, chunk_size=2))

    assert [len(chunk) for chunk in chunks] == [2, 1]
    transactions = [transaction for chunk in chunks for transaction in chunk]
    assert [transaction.source_row_number for transaction in transactions] == [1, 2, 3]
    assert transactions[0].amount == Decimal("0.10")
    assert transactions[0].amount.as_tuple().exponent == -2
    assert isinstance(transactions[0].amount, Decimal)
    assert transactions[0].database_row()[5] is transactions[0].amount
    assert transactions[0].transaction_id != transactions[1].transaction_id


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"amount": "0.001"}, "more than two decimal places"),
        ({"amount": "NaN"}, "finite non-negative"),
        ({"amount": "-1.00"}, "finite non-negative"),
        ({"isFraud": "2"}, "isFraud must be 0 or 1"),
        ({"step": "0"}, "step must be greater than zero"),
        ({"type": "UNKNOWN"}, "unsupported transaction type"),
    ],
)
def test_parse_paysim_row_rejects_invalid_values(overrides, message):
    with pytest.raises(SourceRowError, match=message):
        parse_paysim_row(sample_row(**overrides), 7, DATASET_ID, "paysim.csv")


def test_chunk_size_uses_environment_configuration(monkeypatch):
    monkeypatch.setenv("PAYSIM_INGESTION_CHUNK_SIZE", "1234")
    assert configured_chunk_size() == 1234

    monkeypatch.setenv("PAYSIM_INGESTION_CHUNK_SIZE", "0")
    with pytest.raises(ValueError, match="greater than zero"):
        configured_chunk_size()


def test_summary_keeps_outcomes_distinct_and_reconciled():
    summary = IngestionSummary(DATASET_ID, "paysim.csv")
    summary.record_successful_chunk(chunk_size=10, inserted=7)
    summary.record_failed_records(2)

    assert summary.processed == 12
    assert summary.inserted == 7
    assert summary.skipped == 3
    assert summary.failed == 2
    assert "processed=12, inserted=7, skipped=3, failed=2" in summary.format()


def test_reconcile_ingestion_requires_source_database_and_expected_totals():
    totals = RecordCounts(total=3, fraud=1, flagged_fraud=0)
    summary = IngestionSummary(
        DATASET_ID,
        "paysim.csv",
        processed=3,
        inserted=3,
        source_totals=totals,
        database_totals=totals,
    )

    reconcile_ingestion(summary, expected_totals=totals)
    assert summary.reconciled is True

    summary.reconciled = False
    summary.database_totals = RecordCounts(total=2, fraud=1, flagged_fraud=0)
    with pytest.raises(IngestionRunError, match="database totals"):
        reconcile_ingestion(summary, expected_totals=totals)
