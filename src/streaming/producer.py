"""Replay repository-local PaySim rows as paced, versioned Kafka events."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from confluent_kafka import KafkaError, Producer

from src.data.paysim_ingestion import (
    PaySimTransaction,
    discover_paysim_csv,
    parse_paysim_row,
    validate_source_columns,
)
from src.data.transaction_identity import calculate_dataset_sha256
from src.streaming.event_contract import build_event_payload


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProducerSettings:
    """Validated, environment-independent replay controls."""

    bootstrap_servers: str
    topic: str
    start_row: int = 1
    limit: int = 100
    rate_per_second: float = 10.0


@dataclass(frozen=True)
class ProducerSummary:
    """Broker-confirmed result of one bounded replay."""

    requested: int
    attempted: int
    acknowledged: int
    failed: int


def _positive_integer(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _positive_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def load_producer_settings() -> ProducerSettings:
    """Load replay behavior from environment variables so code changes are unnecessary."""
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TRANSACTION_TOPIC", "paysim.transactions.v1")
    if not bootstrap_servers or not topic:
        raise ValueError("Kafka bootstrap servers and topic must not be empty")
    return ProducerSettings(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        start_row=_positive_integer(os.getenv("PAYSIM_REPLAY_START_ROW", "1"), "start row"),
        limit=_positive_integer(os.getenv("PAYSIM_REPLAY_LIMIT", "100"), "limit"),
        rate_per_second=_positive_float(
            os.getenv("PAYSIM_REPLAY_RATE", "10"), "replay rate"
        ),
    )


def iter_replay_transactions(
    csv_path: Path,
    source_dataset_id: str,
    start_row: int,
    limit: int,
) -> Iterator[PaySimTransaction]:
    """Read only the requested source window while preserving original row identities."""
    if start_row <= 0 or limit <= 0:
        raise ValueError("start_row and limit must be greater than zero")
    end_row = start_row + limit
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        validate_source_columns(reader.fieldnames)
        for row_number, row in enumerate(reader, start=1):
            if row_number < start_row:
                continue
            if row_number >= end_row:
                break
            yield parse_paysim_row(row, row_number, source_dataset_id, csv_path.name)


class ReplayPacer:
    """Space events against a monotonic clock to make the configured rate observable."""

    def __init__(
        self,
        rate_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be greater than zero")
        self._interval = 1 / rate_per_second
        self._clock = clock
        self._sleeper = sleeper
        self._next_send_at: float | None = None

    def wait(self) -> None:
        """Wait only as long as needed to maintain the requested average spacing."""
        now = self._clock()
        if self._next_send_at is None:
            self._next_send_at = now
        delay = self._next_send_at - now
        if delay > 0:
            self._sleeper(delay)
        self._next_send_at = max(self._next_send_at, now) + self._interval


def replay_paysim(
    settings: ProducerSettings,
    csv_path: str | Path | None = None,
    producer_factory: Callable[[dict[str, object]], Producer] = Producer,
) -> ProducerSummary:
    """Publish a bounded source window and require broker acknowledgement for every event."""
    resolved_csv = discover_paysim_csv(csv_path)
    dataset_id = calculate_dataset_sha256(resolved_csv)
    producer = producer_factory(
        {
            "bootstrap.servers": settings.bootstrap_servers,
            "enable.idempotence": True,
            "acks": "all",
            "client.id": "paysim-replay-producer",
        }
    )
    pacer = ReplayPacer(settings.rate_per_second)
    outcomes: list[KafkaError | None] = []
    attempted = 0

    def delivery_report(error: KafkaError | None, _message: object) -> None:
        outcomes.append(error)

    for transaction in iter_replay_transactions(
        resolved_csv, dataset_id, settings.start_row, settings.limit
    ):
        pacer.wait()
        payload = build_event_payload(transaction)
        while True:
            try:
                producer.produce(
                    settings.topic,
                    key=str(transaction.transaction_id),
                    value=payload,
                    on_delivery=delivery_report,
                )
                break
            except BufferError:
                producer.poll(0.1)
        attempted += 1
        producer.poll(0)

    remaining = producer.flush(30)
    if attempted != settings.limit:
        raise RuntimeError(
            f"source ended after {attempted} rows; requested {settings.limit} events"
        )
    if remaining:
        raise RuntimeError(f"{remaining} Kafka events remained unflushed")

    failed = sum(error is not None for error in outcomes)
    summary = ProducerSummary(settings.limit, attempted, len(outcomes) - failed, failed)
    if len(outcomes) != attempted or failed:
        raise RuntimeError(
            "Kafka did not acknowledge every event: "
            f"attempted={attempted}, acknowledged={summary.acknowledged}, failed={failed}"
        )
    LOGGER.info(
        "Replay complete: requested=%s attempted=%s acknowledged=%s failed=%s",
        summary.requested,
        summary.attempted,
        summary.acknowledged,
        summary.failed,
    )
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    """Build a small CLI whose defaults come from the documented environment."""
    defaults = load_producer_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv")
    parser.add_argument("--start-row", type=int, default=defaults.start_row)
    parser.add_argument("--limit", type=int, default=defaults.limit)
    parser.add_argument("--rate", type=float, default=defaults.rate_per_second)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    return parser


def main() -> int:
    """Run the producer CLI."""
    arguments = build_argument_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    defaults = load_producer_settings()
    settings = ProducerSettings(
        defaults.bootstrap_servers,
        defaults.topic,
        arguments.start_row,
        arguments.limit,
        arguments.rate,
    )
    try:
        replay_paysim(settings, arguments.csv)
    except (RuntimeError, ValueError) as error:
        LOGGER.error("PaySim replay failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
