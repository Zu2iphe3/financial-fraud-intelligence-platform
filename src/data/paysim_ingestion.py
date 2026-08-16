"""Validate and ingest the local PaySim CSV into PostgreSQL in bounded chunks."""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterator, Mapping, Sequence
from uuid import UUID

import psycopg

from src.data.postgres import load_postgres_settings
from src.data.transaction_identity import calculate_dataset_sha256, create_transaction_id


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "raw"
DEFAULT_CHUNK_SIZE = 50_000
CHUNK_SIZE_ENVIRONMENT_VARIABLE = "PAYSIM_INGESTION_CHUNK_SIZE"

PAYSIM_COLUMNS = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)
TRANSACTION_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
MONEY_COLUMNS = (
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)

DATABASE_COLUMNS = (
    "transaction_id",
    "source_dataset_id",
    "source_row_number",
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
    "first_ingestion_source",
    "source_file_name",
)


class PaySimIngestionError(Exception):
    """Base error for failures that can be explained safely to an operator."""


class SourceSchemaError(PaySimIngestionError):
    """Raised before ingestion when the CSV header is incompatible."""


class SourceRowError(PaySimIngestionError):
    """Raised when a PaySim row violates the documented source contract."""


class IngestionRunError(PaySimIngestionError):
    """Raised with the partial summary when an active ingestion run fails."""

    def __init__(self, message: str, summary: IngestionSummary):
        super().__init__(message)
        self.summary = summary


@dataclass(frozen=True)
class RecordCounts:
    """Comparable transaction and fraud-label totals."""

    total: int
    fraud: int
    flagged_fraud: int


KNOWN_PAYSIM_TOTALS = RecordCounts(total=6_362_620, fraud=8_213, flagged_fraud=16)


@dataclass(frozen=True)
class PaySimTransaction:
    """One validated PaySim row with exact monetary values and stable identity."""

    transaction_id: UUID
    source_dataset_id: str
    source_row_number: int
    step: int
    transaction_type: str
    amount: Decimal
    name_orig: str
    old_balance_orig: Decimal
    new_balance_orig: Decimal
    name_dest: str
    old_balance_dest: Decimal
    new_balance_dest: Decimal
    is_fraud: int
    is_flagged_fraud: int
    source_file_name: str

    def database_row(self) -> tuple[object, ...]:
        """Return values in the explicit PostgreSQL staging-column order."""
        return (
            self.transaction_id,
            self.source_dataset_id,
            self.source_row_number,
            self.step,
            self.transaction_type,
            self.amount,
            self.name_orig,
            self.old_balance_orig,
            self.new_balance_orig,
            self.name_dest,
            self.old_balance_dest,
            self.new_balance_dest,
            self.is_fraud,
            self.is_flagged_fraud,
            "batch",
            self.source_file_name,
        )


@dataclass
class IngestionSummary:
    """Track mutually exclusive record outcomes for one ingestion run."""

    dataset_id: str
    source_file_name: str
    processed: int = 0
    inserted: int = 0
    skipped: int = 0
    failed: int = 0
    source_totals: RecordCounts | None = None
    database_totals: RecordCounts | None = None
    reconciled: bool = False

    def record_successful_chunk(self, chunk_size: int, inserted: int) -> None:
        """Record a committed chunk while keeping outcomes internally consistent."""
        if not 0 <= inserted <= chunk_size:
            raise ValueError("inserted must be between zero and chunk_size")
        self.processed += chunk_size
        self.inserted += inserted
        self.skipped += chunk_size - inserted

    def record_failed_records(self, record_count: int) -> None:
        """Record rows whose outcome is failure rather than insertion or skipping."""
        if record_count <= 0:
            raise ValueError("record_count must be greater than zero")
        self.processed += record_count
        self.failed += record_count

    def validate_outcomes(self) -> None:
        """Prevent ambiguous summaries where outcome counts do not reconcile."""
        outcome_total = self.inserted + self.skipped + self.failed
        if self.processed != outcome_total:
            raise ValueError(
                "processed must equal inserted + skipped + failed "
                f"({self.processed} != {outcome_total})"
            )

    def format(self) -> str:
        """Return an operator-friendly summary with every outcome shown separately."""
        self.validate_outcomes()
        return (
            f"processed={self.processed:,}, inserted={self.inserted:,}, "
            f"skipped={self.skipped:,}, failed={self.failed:,}, "
            f"reconciled={'yes' if self.reconciled else 'no'}"
        )


def configured_chunk_size() -> int:
    """Load the chunk size from the environment without requiring source changes."""
    raw_value = os.getenv(CHUNK_SIZE_ENVIRONMENT_VARIABLE, str(DEFAULT_CHUNK_SIZE))
    try:
        chunk_size = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{CHUNK_SIZE_ENVIRONMENT_VARIABLE} must be an integer") from error
    if chunk_size <= 0:
        raise ValueError(f"{CHUNK_SIZE_ENVIRONMENT_VARIABLE} must be greater than zero")
    return chunk_size


def discover_paysim_csv(explicit_path: str | Path | None = None) -> Path:
    """Find one CSV using repository-relative defaults rather than a machine path."""
    if explicit_path is not None:
        requested_path = Path(explicit_path)
        candidate = requested_path if requested_path.is_absolute() else REPOSITORY_ROOT / requested_path
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise PaySimIngestionError(f"PaySim CSV does not exist: {explicit_path}")
        if candidate.suffix.lower() != ".csv":
            raise PaySimIngestionError(f"PaySim source must be a CSV file: {explicit_path}")
        return candidate

    candidates = sorted(RAW_DATA_DIRECTORY.glob("*.csv"))
    if not candidates:
        raise PaySimIngestionError(
            "No PaySim CSV found under data/raw/. Follow the dataset setup in README.md."
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise PaySimIngestionError(
            f"Multiple CSV files found under data/raw/: {names}. Select one with --csv."
        )
    return candidates[0].resolve()


def validate_source_columns(columns: Sequence[str] | None) -> None:
    """Reject an incompatible source before any database record is attempted."""
    actual_columns = tuple(columns or ())
    missing = [column for column in PAYSIM_COLUMNS if column not in actual_columns]
    unexpected = [column for column in actual_columns if column not in PAYSIM_COLUMNS]
    order_changed = not missing and not unexpected and actual_columns != PAYSIM_COLUMNS

    problems = []
    if missing:
        problems.append(f"missing columns: {', '.join(missing)}")
    if unexpected:
        problems.append(f"unexpected columns: {', '.join(unexpected)}")
    if order_changed:
        problems.append("columns are not in the documented PaySim order")
    if problems:
        raise SourceSchemaError("Incompatible PaySim schema; " + "; ".join(problems))


def validate_csv_header(csv_path: Path) -> None:
    """Read only the header so schema failures happen before ingestion starts."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.reader(source_file)
        validate_source_columns(next(reader, None))


def _parse_integer(raw_value: str, column: str, row_number: int) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as error:
        raise SourceRowError(f"Row {row_number}: {column} must be an integer") from error
    return value


def _parse_money(raw_value: str, column: str, row_number: int) -> Decimal:
    """Parse CSV text directly as Decimal so binary floating-point never alters money."""
    try:
        value = Decimal(raw_value)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise SourceRowError(f"Row {row_number}: {column} must be a decimal number") from error

    if not value.is_finite() or value < 0:
        raise SourceRowError(f"Row {row_number}: {column} must be a finite non-negative value")
    if value.as_tuple().exponent < -2:
        raise SourceRowError(f"Row {row_number}: {column} cannot have more than two decimal places")
    integer_digits = max(value.adjusted() + 1, 1) if value else 1
    if integer_digits > 18:
        raise SourceRowError(f"Row {row_number}: {column} exceeds NUMERIC(20,2)")
    return value


def _parse_label(raw_value: str, column: str, row_number: int) -> int:
    value = _parse_integer(raw_value, column, row_number)
    if value not in (0, 1):
        raise SourceRowError(f"Row {row_number}: {column} must be 0 or 1")
    return value


def parse_paysim_row(
    row: Mapping[str, str],
    row_number: int,
    source_dataset_id: str,
    source_file_name: str,
) -> PaySimTransaction:
    """Validate one source row and preserve its exact source meaning."""
    step = _parse_integer(row["step"], "step", row_number)
    if step <= 0:
        raise SourceRowError(f"Row {row_number}: step must be greater than zero")

    transaction_type = row["type"]
    if transaction_type not in TRANSACTION_TYPES:
        raise SourceRowError(f"Row {row_number}: unsupported transaction type {transaction_type!r}")
    if not row["nameOrig"] or not row["nameDest"]:
        raise SourceRowError(f"Row {row_number}: origin and destination names are required")

    monetary_values = {
        column: _parse_money(row[column], column, row_number) for column in MONEY_COLUMNS
    }
    return PaySimTransaction(
        transaction_id=create_transaction_id(source_dataset_id, row_number),
        source_dataset_id=source_dataset_id,
        source_row_number=row_number,
        step=step,
        transaction_type=transaction_type,
        amount=monetary_values["amount"],
        name_orig=row["nameOrig"],
        old_balance_orig=monetary_values["oldbalanceOrg"],
        new_balance_orig=monetary_values["newbalanceOrig"],
        name_dest=row["nameDest"],
        old_balance_dest=monetary_values["oldbalanceDest"],
        new_balance_dest=monetary_values["newbalanceDest"],
        is_fraud=_parse_label(row["isFraud"], "isFraud", row_number),
        is_flagged_fraud=_parse_label(row["isFlaggedFraud"], "isFlaggedFraud", row_number),
        source_file_name=source_file_name,
    )


def iter_paysim_chunks(
    csv_path: Path,
    source_dataset_id: str,
    chunk_size: int,
) -> Iterator[list[PaySimTransaction]]:
    """Yield validated chunks while retaining original one-based data-row numbers."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        validate_source_columns(reader.fieldnames)
        chunk: list[PaySimTransaction] = []
        for row_number, row in enumerate(reader, start=1):
            chunk.append(
                parse_paysim_row(row, row_number, source_dataset_id, csv_path.name)
            )
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def create_staging_table(connection: psycopg.Connection) -> None:
    """Create one session-local buffer used for fast COPY plus safe conflict handling."""
    column_list = ", ".join(DATABASE_COLUMNS)
    with connection.transaction():
        connection.execute(
            f"""
            CREATE TEMP TABLE IF NOT EXISTS paysim_ingestion_stage
                ON COMMIT DELETE ROWS
                AS SELECT {column_list} FROM raw.transactions WITH NO DATA
            """
        )


def persist_chunk(connection: psycopg.Connection, chunk: Sequence[PaySimTransaction]) -> int:
    """Atomically stage and insert one chunk, returning the number newly inserted."""
    if not chunk:
        return 0

    column_list = ", ".join(DATABASE_COLUMNS)
    copy_statement = f"COPY paysim_ingestion_stage ({column_list}) FROM STDIN"
    insert_statement = f"""
        INSERT INTO raw.transactions ({column_list})
        SELECT {column_list}
        FROM paysim_ingestion_stage
        ON CONFLICT DO NOTHING
    """

    # COPY provides bulk performance, while the surrounding transaction ensures that a
    # partially failed chunk never becomes a reported or committed partial success.
    with connection.transaction():
        with connection.cursor() as cursor:
            with cursor.copy(copy_statement) as copy:
                for transaction in chunk:
                    copy.write_row(transaction.database_row())
            cursor.execute(insert_statement)
            inserted = cursor.rowcount
    return inserted


def safe_database_error(error: Exception) -> str:
    """Describe a database failure without logging PostgreSQL's rejected-row detail."""
    if isinstance(error, psycopg.Error):
        primary_message = error.diag.message_primary
        if primary_message:
            return f"{type(error).__name__}: {primary_message}"
    return type(error).__name__


def count_chunk(chunk: Sequence[PaySimTransaction]) -> RecordCounts:
    """Return source totals for reconciliation without retaining prior chunks."""
    return RecordCounts(
        total=len(chunk),
        fraud=sum(transaction.is_fraud for transaction in chunk),
        flagged_fraud=sum(transaction.is_flagged_fraud for transaction in chunk),
    )


def add_counts(left: RecordCounts, right: RecordCounts) -> RecordCounts:
    """Combine bounded per-chunk counters into complete source totals."""
    return RecordCounts(
        total=left.total + right.total,
        fraud=left.fraud + right.fraud,
        flagged_fraud=left.flagged_fraud + right.flagged_fraud,
    )


def database_counts(connection: psycopg.Connection, source_dataset_id: str) -> RecordCounts:
    """Count only the exact dataset being reconciled, excluding unrelated records."""
    row = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(is_fraud), 0), COALESCE(SUM(is_flagged_fraud), 0)
        FROM raw.transactions
        WHERE source_dataset_id = %s
        """,
        (source_dataset_id,),
    ).fetchone()
    return RecordCounts(total=row[0], fraud=row[1], flagged_fraud=row[2])


def reconcile_ingestion(
    summary: IngestionSummary,
    expected_totals: RecordCounts | None = KNOWN_PAYSIM_TOTALS,
) -> None:
    """Fail when known, source, or database totals do not describe the same dataset."""
    if summary.source_totals is None or summary.database_totals is None:
        raise ValueError("source and database totals are required for reconciliation")

    problems = []
    if expected_totals is not None and summary.source_totals != expected_totals:
        problems.append(
            f"source totals {summary.source_totals} do not match expected {expected_totals}"
        )
    if summary.database_totals != summary.source_totals:
        problems.append(
            f"database totals {summary.database_totals} do not match source "
            f"{summary.source_totals}"
        )
    if problems:
        raise IngestionRunError(
            "Reconciliation failed: " + "; ".join(problems) + f"; {summary.format()}",
            summary,
        )
    summary.reconciled = True


def ingest_paysim(
    csv_path: str | Path | None = None,
    chunk_size: int | None = None,
    expected_totals: RecordCounts | None = KNOWN_PAYSIM_TOTALS,
) -> IngestionSummary:
    """Run validated, idempotent PaySim ingestion and return a reconciled summary."""
    resolved_csv = discover_paysim_csv(csv_path)
    validate_csv_header(resolved_csv)
    resolved_chunk_size = configured_chunk_size() if chunk_size is None else chunk_size
    if resolved_chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    LOGGER.info("Calculating exact dataset fingerprint for %s", resolved_csv.name)
    source_dataset_id = calculate_dataset_sha256(resolved_csv)
    summary = IngestionSummary(source_dataset_id, resolved_csv.name)
    source_totals = RecordCounts(total=0, fraud=0, flagged_fraud=0)
    settings = load_postgres_settings()

    LOGGER.info(
        "Starting PaySim ingestion: file=%s, chunk_size=%s, dataset_id=%s",
        resolved_csv.name,
        f"{resolved_chunk_size:,}",
        source_dataset_id,
    )
    with psycopg.connect(**settings.connection_kwargs()) as connection:
        create_staging_table(connection)
        try:
            for chunk_number, chunk in enumerate(
                iter_paysim_chunks(resolved_csv, source_dataset_id, resolved_chunk_size),
                start=1,
            ):
                source_totals = add_counts(source_totals, count_chunk(chunk))
                try:
                    inserted = persist_chunk(connection, chunk)
                except Exception as error:
                    summary.record_failed_records(len(chunk))
                    safe_cause = safe_database_error(error)
                    LOGGER.error(
                        "Database batch failed: chunk=%s, rows=%s, cause=%s",
                        chunk_number,
                        f"{len(chunk):,}",
                        safe_cause,
                    )
                    raise IngestionRunError(
                        f"Database batch {chunk_number} failed ({safe_cause}); {summary.format()}",
                        summary,
                    ) from error

                summary.record_successful_chunk(len(chunk), inserted)
                LOGGER.info("Chunk %s committed: %s", chunk_number, summary.format())
        except SourceRowError as error:
            summary.record_failed_records(1)
            LOGGER.error("Source validation failed: %s", error)
            raise IngestionRunError(f"{error}; {summary.format()}", summary) from error

        summary.source_totals = source_totals
        summary.database_totals = database_counts(connection, source_dataset_id)
        reconcile_ingestion(summary, expected_totals)

    LOGGER.info("PaySim ingestion completed: %s", summary.format())
    LOGGER.info(
        "Reconciled totals: transactions=%s, fraud=%s, flagged_fraud=%s",
        f"{summary.database_totals.total:,}",
        f"{summary.database_totals.fraud:,}",
        f"{summary.database_totals.flagged_fraud:,}",
    )
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the documented command-line interface for local batch ingestion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        help="Repository-relative PaySim CSV path; defaults to the only CSV under data/raw/.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Rows per database batch; defaults to PAYSIM_INGESTION_CHUNK_SIZE "
            f"or {DEFAULT_CHUNK_SIZE:,}."
        ),
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    return parser


def main() -> int:
    """Run the ingestion CLI and return a shell-friendly exit status."""
    arguments = build_argument_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        summary = ingest_paysim(arguments.csv, arguments.chunk_size)
    except (PaySimIngestionError, ValueError) as error:
        LOGGER.error("PaySim ingestion failed: %s", error)
        return 1
    LOGGER.info("Final outcome: %s", summary.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
