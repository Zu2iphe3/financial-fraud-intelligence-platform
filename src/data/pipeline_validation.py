"""Post-ingestion quality gate for the orchestrated PaySim batch pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from src.data.paysim_ingestion import IngestionSummary, RecordCounts


class PipelineValidationError(RuntimeError):
    """Raised when persisted PaySim data is unsafe for downstream dbt models."""


@dataclass(frozen=True)
class PipelineValidationResult:
    """Small, log-safe summary of the database quality checks."""

    dataset_id: str
    database_totals: RecordCounts
    duplicate_transaction_ids: int
    duplicate_source_rows: int


def _count_duplicate_groups(
    connection: psycopg.Connection,
    source_dataset_id: str,
    columns: str,
) -> int:
    """Count duplicate identity groups even if database constraints later change."""
    row = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT {columns}
            FROM raw.transactions
            WHERE source_dataset_id = %s
            GROUP BY {columns}
            HAVING COUNT(*) > 1
        ) AS duplicate_groups
        """,
        (source_dataset_id,),
    ).fetchone()
    return int(row[0])


def validate_pipeline_state(
    connection: psycopg.Connection,
    summary: IngestionSummary,
    expected_totals: RecordCounts,
) -> PipelineValidationResult:
    """Block analytics when ingestion outcomes or persisted totals do not reconcile.

    This gate deliberately runs after ingestion and before dbt so a partial, duplicated,
    or unexpected PaySim load cannot silently become trusted analytics output.
    """
    summary.validate_outcomes()

    row = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(is_fraud), 0), COALESCE(SUM(is_flagged_fraud), 0)
        FROM raw.transactions
        WHERE source_dataset_id = %s
        """,
        (summary.dataset_id,),
    ).fetchone()
    database_totals = RecordCounts(
        total=int(row[0]), fraud=int(row[1]), flagged_fraud=int(row[2])
    )
    duplicate_transaction_ids = _count_duplicate_groups(
        connection, summary.dataset_id, "transaction_id"
    )
    duplicate_source_rows = _count_duplicate_groups(
        connection, summary.dataset_id, "source_dataset_id, source_row_number"
    )

    problems: list[str] = []
    if summary.failed:
        problems.append(f"ingestion reported {summary.failed:,} failed records")
    if not summary.reconciled:
        problems.append("ingestion did not report a reconciled result")
    if summary.source_totals != expected_totals:
        problems.append(
            f"source totals {summary.source_totals} do not match expected {expected_totals}"
        )
    if database_totals != expected_totals:
        problems.append(
            f"database totals {database_totals} do not match expected {expected_totals}"
        )
    if duplicate_transaction_ids:
        problems.append(
            f"found {duplicate_transaction_ids:,} duplicate transaction identity groups"
        )
    if duplicate_source_rows:
        problems.append(f"found {duplicate_source_rows:,} duplicate source-row groups")

    if problems:
        raise PipelineValidationError("Post-ingestion validation failed: " + "; ".join(problems))

    return PipelineValidationResult(
        dataset_id=summary.dataset_id,
        database_totals=database_totals,
        duplicate_transaction_ids=duplicate_transaction_ids,
        duplicate_source_rows=duplicate_source_rows,
    )
