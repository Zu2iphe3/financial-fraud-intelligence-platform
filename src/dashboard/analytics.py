"""Read-only access to trusted dbt marts for dashboard analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from src.data.postgres import PostgresSettings


TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")


class AnalyticsUnavailableError(RuntimeError):
    """Raised when the dashboard cannot read its trusted analytics marts."""


@dataclass(frozen=True)
class AnalyticsFilters:
    """Supported read-only filters for the historical analytics experience."""

    start_day: int
    end_day: int
    transaction_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.start_day < 1 or self.end_day < self.start_day:
            raise ValueError("Simulation-day range is invalid")
        unknown = set(self.transaction_types) - set(TRANSACTION_TYPES)
        if unknown:
            raise ValueError("Transaction-type filter contains unsupported values")


@dataclass(frozen=True)
class AnalyticsMetrics:
    transaction_count: int
    fraud_count: int
    fraud_rate: float
    fraud_amount: float


@dataclass(frozen=True)
class AnalyticsSnapshot:
    """All filtered values rendered together so cards and charts reconcile."""

    metrics: AnalyticsMetrics
    daily: pd.DataFrame
    by_transaction_type: pd.DataFrame


HEADLINE_SQL = """
/* dashboard_headline */
select
    count(*)::bigint as transaction_count,
    count(*) filter (where is_fraudulent)::bigint as fraud_count,
    coalesce(sum(amount) filter (where is_fraudulent), 0)::numeric as fraud_amount,
    coalesce(
        count(*) filter (where is_fraudulent)::numeric / nullif(count(*), 0), 0
    )::numeric as fraud_rate
from analytics_marts.fct_transactions
{where_clause}
"""

DAILY_SQL = """
/* dashboard_daily */
with filtered_transactions as (
    select simulation_day, simulation_hour_of_day, is_fraudulent
    from analytics_marts.fct_transactions
    {where_clause}
)
select
    filtered.simulation_day,
    count(*)::bigint as transaction_count,
    count(*) filter (where filtered.is_fraudulent)::bigint as fraud_count,
    coalesce(
        count(*) filter (where filtered.is_fraudulent)::numeric / nullif(count(*), 0), 0
    )::numeric as fraud_rate,
    bool_or(day_mart.is_partial_day) as is_partial_day
from filtered_transactions as filtered
join analytics_marts.mart_fraud_by_day as day_mart
  on day_mart.simulation_day = filtered.simulation_day
group by filtered.simulation_day
order by filtered.simulation_day
"""

TRANSACTION_TYPE_SQL = """
/* dashboard_transaction_type */
select
    transaction_type,
    count(*)::bigint as transaction_count,
    count(*) filter (where is_fraudulent)::bigint as fraud_count,
    coalesce(
        count(*) filter (where is_fraudulent)::numeric / nullif(count(*), 0), 0
    )::numeric as fraud_rate
from analytics_marts.fct_transactions
{where_clause}
group by transaction_type
order by transaction_type
"""

FILTER_OPTIONS_SQL = """
/* dashboard_filter_options */
select
    (select min(simulation_day) from analytics_marts.mart_fraud_by_day) as min_day,
    (select max(simulation_day) from analytics_marts.mart_fraud_by_day) as max_day,
    (select array_agg(transaction_type order by transaction_type)
       from analytics_marts.mart_fraud_by_transaction_type) as transaction_types
"""


def _default_connection_factory(**kwargs: Any):
    return psycopg.connect(**kwargs)


class AnalyticsRepository:
    """Query dbt marts without exposing raw data or mutating source relations."""

    def __init__(
        self,
        settings: PostgresSettings,
        connection_factory: Callable[..., Any] = _default_connection_factory,
    ) -> None:
        self.settings = settings
        self.connection_factory = connection_factory

    def _connect(self):
        return self.connection_factory(
            **self.settings.connection_kwargs(), connect_timeout=3, row_factory=dict_row
        )

    @staticmethod
    def _where(filters: AnalyticsFilters) -> tuple[str, tuple[Any, ...]]:
        clauses = ["simulation_day between %s and %s"]
        parameters: list[Any] = [filters.start_day, filters.end_day]
        if filters.transaction_types:
            clauses.append("transaction_type = any(%s)")
            parameters.append(list(filters.transaction_types))
        return "where " + " and ".join(clauses), tuple(parameters)

    def load_filter_options(self) -> tuple[int, int, tuple[str, ...]]:
        """Read available filters from aggregate marts rather than hard-code days."""

        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(FILTER_OPTIONS_SQL)
                row = cursor.fetchone()
        except Exception as error:
            raise AnalyticsUnavailableError("Historical analytics are unavailable") from error
        if not row or row["min_day"] is None or row["max_day"] is None:
            raise AnalyticsUnavailableError("Analytics marts contain no transactions")
        return (
            int(row["min_day"]),
            int(row["max_day"]),
            tuple(str(item) for item in row["transaction_types"]),
        )

    def load_snapshot(self, filters: AnalyticsFilters) -> AnalyticsSnapshot:
        """Load cards and charts under one filter contract using SELECT queries only."""

        where_clause, parameters = self._where(filters)
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(HEADLINE_SQL.format(where_clause=where_clause), parameters)
                headline = cursor.fetchone()
                cursor.execute(DAILY_SQL.format(where_clause=where_clause), parameters)
                daily_rows = cursor.fetchall()
                cursor.execute(
                    TRANSACTION_TYPE_SQL.format(where_clause=where_clause), parameters
                )
                type_rows = cursor.fetchall()
        except Exception as error:
            raise AnalyticsUnavailableError("Historical analytics are unavailable") from error

        headline = headline or {}
        metrics = AnalyticsMetrics(
            transaction_count=int(headline.get("transaction_count", 0)),
            fraud_count=int(headline.get("fraud_count", 0)),
            fraud_rate=float(headline.get("fraud_rate", 0)),
            fraud_amount=float(headline.get("fraud_amount", 0)),
        )
        return AnalyticsSnapshot(
            metrics=metrics,
            daily=pd.DataFrame(daily_rows),
            by_transaction_type=pd.DataFrame(type_rows),
        )
