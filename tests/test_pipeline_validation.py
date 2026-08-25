"""Unit tests for the post-ingestion Airflow quality gate."""

from unittest.mock import MagicMock

import pytest

from src.data.paysim_ingestion import IngestionSummary, RecordCounts
from src.data.pipeline_validation import PipelineValidationError, validate_pipeline_state


EXPECTED = RecordCounts(total=10, fraud=2, flagged_fraud=1)


def _summary(**overrides) -> IngestionSummary:
    values = {
        "dataset_id": "a" * 64,
        "source_file_name": "paysim.csv",
        "processed": 10,
        "inserted": 10,
        "skipped": 0,
        "failed": 0,
        "source_totals": EXPECTED,
        "database_totals": EXPECTED,
        "reconciled": True,
    }
    values.update(overrides)
    return IngestionSummary(**values)


def _connection(database_counts=(10, 2, 1), duplicates=(0, 0)):
    connection = MagicMock()
    results = [database_counts, (duplicates[0],), (duplicates[1],)]
    query_results = []
    for result in results:
        cursor = MagicMock()
        cursor.fetchone.return_value = result
        query_results.append(cursor)
    connection.execute.side_effect = query_results
    return connection


def test_validate_pipeline_state_accepts_reconciled_unique_data():
    result = validate_pipeline_state(_connection(), _summary(), EXPECTED)

    assert result.database_totals == EXPECTED
    assert result.duplicate_transaction_ids == 0
    assert result.duplicate_source_rows == 0


@pytest.mark.parametrize(
    ("connection", "summary", "message"),
    [
        (_connection(database_counts=(9, 2, 1)), _summary(), "database totals"),
        (_connection(duplicates=(1, 0)), _summary(), "duplicate transaction"),
        (_connection(duplicates=(0, 1)), _summary(), "duplicate source-row"),
        (_connection(), _summary(reconciled=False), "did not report a reconciled"),
        (
            _connection(),
            _summary(processed=10, inserted=9, failed=1, reconciled=False),
            "failed records",
        ),
    ],
)
def test_validate_pipeline_state_rejects_unsafe_data(connection, summary, message):
    with pytest.raises(PipelineValidationError, match=message):
        validate_pipeline_state(connection, summary, EXPECTED)


def test_validate_pipeline_state_rejects_inconsistent_outcome_summary():
    with pytest.raises(ValueError, match="processed must equal"):
        validate_pipeline_state(_connection(), _summary(inserted=9), EXPECTED)
