"""Tests for read-only trusted-mart dashboard analytics."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from src.dashboard.analytics import (
    AnalyticsFilters,
    AnalyticsRepository,
    AnalyticsUnavailableError,
)
from src.data.postgres import PostgresSettings
from src.dashboard.pages.overview import (
    AGGREGATE_TABLE_COLUMN_WIDTHS,
    DAILY_TABLE_HEIGHT,
    DAILY_TABLE_VISIBLE_ROWS,
    TABLE_HEADER_HEIGHT,
    TABLE_ROW_HEIGHT,
    TYPE_TABLE_HEIGHT,
    TYPE_TABLE_VISIBLE_ROWS,
    _aggregated_display_tables,
    _compact_number,
)


class FakeCursor:
    def __init__(self) -> None:
        self.query = ""
        self.executions = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, parameters=()) -> None:
        self.query = query
        self.executions.append((query, parameters))

    def fetchone(self):
        if "dashboard_filter_options" in self.query:
            return {
                "min_day": 1,
                "max_day": 31,
                "transaction_types": ["CASH_IN", "CASH_OUT", "TRANSFER"],
            }
        return {
            "transaction_count": 100,
            "fraud_count": 4,
            "fraud_amount": Decimal("250.50"),
            "fraud_rate": Decimal("0.04"),
        }

    def fetchall(self):
        if "dashboard_daily" in self.query:
            return [
                {
                    "simulation_day": 2,
                    "transaction_count": 100,
                    "fraud_count": 4,
                    "fraud_rate": Decimal("0.04"),
                    "is_partial_day": False,
                }
            ]
        return [
            {
                "transaction_type": "TRANSFER",
                "transaction_count": 100,
                "fraud_count": 4,
                "fraud_rate": Decimal("0.04"),
            }
        ]


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.fake_cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self.fake_cursor


def _settings() -> PostgresSettings:
    return PostgresSettings("fraud", "user", "secret", "localhost", 5432)


def test_analytics_filters_query_only_trusted_marts_with_parameters() -> None:
    cursor = FakeCursor()
    repository = AnalyticsRepository(
        _settings(), connection_factory=lambda **_: FakeConnection(cursor)
    )

    assert repository.load_filter_options() == (
        1,
        31,
        ("CASH_IN", "CASH_OUT", "TRANSFER"),
    )
    snapshot = repository.load_snapshot(AnalyticsFilters(2, 10, ("TRANSFER",)))

    assert snapshot.metrics.transaction_count == 100
    assert snapshot.metrics.fraud_rate == 0.04
    assert snapshot.daily.iloc[0]["simulation_day"] == 2
    assert snapshot.by_transaction_type.iloc[0]["transaction_type"] == "TRANSFER"
    assert all("raw.transactions" not in query for query, _ in cursor.executions)
    assert all(query.lstrip().startswith("/* dashboard_") for query, _ in cursor.executions)
    daily_query = next(query for query, _ in cursor.executions if "dashboard_daily" in query)
    assert "mart_fraud_by_day" in daily_query
    assert "bool_or(day_mart.is_partial_day)" in daily_query
    filtered_parameters = [parameters for _, parameters in cursor.executions if parameters]
    assert filtered_parameters == [(2, 10, ["TRANSFER"])] * 3


def test_analytics_repository_returns_controlled_unavailable_error() -> None:
    def fail(**_):
        raise OSError("private database details")

    repository = AnalyticsRepository(_settings(), connection_factory=fail)

    with pytest.raises(AnalyticsUnavailableError) as captured:
        repository.load_filter_options()

    assert str(captured.value) == "Historical analytics are unavailable"
    assert "private" not in str(captured.value)


@pytest.mark.parametrize(
    "filters",
    [AnalyticsFilters(1, 31), AnalyticsFilters(1, 31, ("CASH_OUT", "TRANSFER"))],
)
def test_filter_contract_accepts_supported_ranges(filters) -> None:
    assert filters.start_day == 1


def test_filter_contract_rejects_unknown_transaction_type() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        AnalyticsFilters(1, 31, ("UNKNOWN",))


def test_overview_formats_large_fraud_value_without_changing_source_value() -> None:
    fraud_value = 12_056_415_427.84

    assert _compact_number(fraud_value) == "12.06B"
    assert fraud_value == 12_056_415_427.84


def test_overview_aggregates_use_readable_labels_without_duplicate_rate() -> None:
    daily = pd.DataFrame(
        [
            {
                "simulation_day": 1,
                "transaction_count": 100,
                "fraud_count": 2,
                "fraud_rate": 0.02,
                "fraud_rate_percent": 2.0,
                "is_partial_day": False,
            }
        ]
    )
    by_type = pd.DataFrame(
        [
            {
                "transaction_type": "TRANSFER",
                "transaction_count": 25,
                "fraud_count": 2,
                "fraud_rate": 0.08,
                "fraud_rate_percent": 8.0,
            }
        ]
    )

    daily_display, type_display = _aggregated_display_tables(daily, by_type)

    assert list(daily_display.columns) == [
        "Day",
        "Transactions",
        "Labelled fraud cases",
        "Labelled fraud rate (%)",
        "Partial day",
    ]
    assert list(type_display.columns) == [
        "Transaction type",
        "Transactions",
        "Labelled fraud cases",
        "Labelled fraud rate (%)",
    ]
    assert daily_display.iloc[0].to_dict() == {
        "Day": 1,
        "Transactions": 100,
        "Labelled fraud cases": 2,
        "Labelled fraud rate (%)": 2.0,
        "Partial day": False,
    }
    assert type_display.iloc[0].to_dict() == {
        "Transaction type": "TRANSFER",
        "Transactions": 25,
        "Labelled fraud cases": 2,
        "Labelled fraud rate (%)": 8.0,
    }


def test_overview_table_heights_fit_complete_rows() -> None:
    assert AGGREGATE_TABLE_COLUMN_WIDTHS == (1, 1)
    assert DAILY_TABLE_HEIGHT == (
        TABLE_HEADER_HEIGHT + TABLE_ROW_HEIGHT * DAILY_TABLE_VISIBLE_ROWS
    )
    assert TYPE_TABLE_HEIGHT == (
        TABLE_HEADER_HEIGHT + TABLE_ROW_HEIGHT * TYPE_TABLE_VISIBLE_ROWS
    )
