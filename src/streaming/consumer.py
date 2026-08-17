"""Validate PaySim Kafka events and persist them before acknowledging offsets."""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable

import psycopg
from confluent_kafka import Consumer, KafkaError, KafkaException, Message

from src.data.paysim_ingestion import DATABASE_COLUMNS, PaySimTransaction
from src.data.postgres import PostgresSettings, load_postgres_settings
from src.streaming.event_contract import EventValidationError, PaySimEvent, parse_event_payload


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsumerSettings:
    """Validated Kafka and database-retry behavior."""

    bootstrap_servers: str
    topic: str
    group_id: str
    database_retry_attempts: int = 3
    database_retry_backoff_seconds: float = 2.0


@dataclass(frozen=True)
class KafkaPosition:
    """The broker position retained as ingestion audit metadata."""

    topic: str
    partition: int
    offset: int


@dataclass(frozen=True)
class PersistenceOutcome:
    """Whether this delivery created a logical transaction and an audit record."""

    transaction_inserted: bool
    event_inserted: bool


@dataclass
class ConsumerSummary:
    """Mutually understandable outcomes for one consumer session."""

    received: int = 0
    valid: int = 0
    invalid: int = 0
    transactions_inserted: int = 0
    events_inserted: int = 0
    duplicate_events: int = 0
    acknowledged: int = 0


def _positive_integer(raw_value: str, name: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _non_negative_float(raw_value: str, name: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def load_consumer_settings() -> ConsumerSettings:
    """Load operational settings without embedding local addresses or credentials."""
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TRANSACTION_TOPIC", "paysim.transactions.v1")
    group_id = os.getenv("KAFKA_CONSUMER_GROUP", "fraud-platform-postgres-v1")
    if not bootstrap_servers or not topic or not group_id:
        raise ValueError("Kafka bootstrap servers, topic, and consumer group must not be empty")
    return ConsumerSettings(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        group_id=group_id,
        database_retry_attempts=_positive_integer(
            os.getenv("KAFKA_DATABASE_RETRY_ATTEMPTS", "3"), "database retry attempts"
        ),
        database_retry_backoff_seconds=_non_negative_float(
            os.getenv("KAFKA_DATABASE_RETRY_BACKOFF_SECONDS", "2"),
            "database retry backoff",
        ),
    )


def _stored_source_values(transaction: PaySimTransaction) -> tuple[object, ...]:
    return (
        transaction.source_dataset_id,
        transaction.source_row_number,
        transaction.step,
        transaction.transaction_type,
        transaction.amount,
        transaction.name_orig,
        transaction.old_balance_orig,
        transaction.new_balance_orig,
        transaction.name_dest,
        transaction.old_balance_dest,
        transaction.new_balance_dest,
        transaction.is_fraud,
        transaction.is_flagged_fraud,
    )


def persist_event(
    connection: psycopg.Connection,
    event: PaySimEvent,
    position: KafkaPosition,
) -> PersistenceOutcome:
    """Atomically store one logical transaction and its first Kafka delivery metadata."""
    columns = ", ".join(DATABASE_COLUMNS)
    placeholders = ", ".join(["%s"] * len(DATABASE_COLUMNS))
    transaction = event.transaction

    with connection.transaction():
        inserted_transaction = connection.execute(
            f"""
            INSERT INTO raw.transactions ({columns})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
            RETURNING transaction_id
            """,
            transaction.database_row("kafka"),
        ).fetchone()

        # A stable ID must never quietly refer to altered transaction content.
        stored = connection.execute(
            """
            SELECT source_dataset_id, source_row_number, step, transaction_type, amount,
                   name_orig, old_balance_orig, new_balance_orig, name_dest,
                   old_balance_dest, new_balance_dest, is_fraud, is_flagged_fraud
            FROM raw.transactions
            WHERE transaction_id = %s
            """,
            (transaction.transaction_id,),
        ).fetchone()
        if stored != _stored_source_values(transaction):
            raise ValueError("transaction_id already exists with different PaySim values")

        inserted_event = connection.execute(
            """
            INSERT INTO raw.transaction_ingestions (
                event_id, transaction_id, event_version, kafka_topic,
                kafka_partition, kafka_offset, published_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (event_id) DO NOTHING
            RETURNING event_id
            """,
            (
                event.event_id,
                transaction.transaction_id,
                event.event_version,
                position.topic,
                position.partition,
                position.offset,
                event.published_at,
            ),
        ).fetchone()

        if inserted_event is None:
            existing_event = connection.execute(
                """
                SELECT transaction_id, event_version
                FROM raw.transaction_ingestions
                WHERE event_id = %s
                """,
                (event.event_id,),
            ).fetchone()
            if existing_event != (transaction.transaction_id, event.event_version):
                raise ValueError("event_id already exists with different event metadata")

    return PersistenceOutcome(inserted_transaction is not None, inserted_event is not None)


def persist_event_with_retry(
    event: PaySimEvent,
    position: KafkaPosition,
    postgres_settings: PostgresSettings,
    attempts: int,
    backoff_seconds: float,
    connect: Callable[..., psycopg.Connection] = psycopg.connect,
    sleeper: Callable[[float], None] = time.sleep,
) -> PersistenceOutcome:
    """Retry temporary database failures without acknowledging the Kafka message."""
    for attempt in range(1, attempts + 1):
        try:
            with connect(**postgres_settings.connection_kwargs()) as connection:
                return persist_event(connection, event, position)
        except psycopg.Error as error:
            LOGGER.warning(
                "Database persistence failed: event_id=%s attempt=%s/%s cause=%s",
                event.event_id,
                attempt,
                attempts,
                type(error).__name__,
            )
            if attempt == attempts:
                raise
            sleeper(backoff_seconds * attempt)
    raise AssertionError("retry loop completed without returning or raising")


def process_message(
    consumer: Consumer,
    message: Message,
    settings: ConsumerSettings,
    postgres_settings: PostgresSettings,
    summary: ConsumerSummary,
    persistence: Callable[..., PersistenceOutcome] = persist_event_with_retry,
) -> None:
    """Process one delivery and commit its offset only after the chosen safe outcome."""
    position = KafkaPosition(message.topic(), message.partition(), message.offset())
    summary.received += 1
    try:
        event = parse_event_payload(message.value())
    except EventValidationError as error:
        # The payload is intentionally omitted so rejected data and labels do not leak to logs.
        LOGGER.error(
            "Invalid Kafka event: topic=%s partition=%s offset=%s validation=%s",
            position.topic,
            position.partition,
            position.offset,
            error,
        )
        summary.invalid += 1
        consumer.commit(message=message, asynchronous=False)
        summary.acknowledged += 1
        return

    outcome = persistence(
        event,
        position,
        postgres_settings,
        settings.database_retry_attempts,
        settings.database_retry_backoff_seconds,
    )
    summary.valid += 1
    summary.transactions_inserted += int(outcome.transaction_inserted)
    summary.events_inserted += int(outcome.event_inserted)
    summary.duplicate_events += int(not outcome.event_inserted)

    # This ordering provides at-least-once delivery: a crash can repeat a safe DB write,
    # but can never acknowledge an event whose database transaction did not commit.
    consumer.commit(message=message, asynchronous=False)
    summary.acknowledged += 1


def consume_events(
    settings: ConsumerSettings,
    max_messages: int | None = None,
    consumer_factory: Callable[[dict[str, object]], Consumer] = Consumer,
) -> ConsumerSummary:
    """Consume until interrupted or until a requested validation count is reached."""
    if max_messages is not None and max_messages <= 0:
        raise ValueError("max_messages must be greater than zero")
    postgres_settings = load_postgres_settings()
    consumer = consumer_factory(
        {
            "bootstrap.servers": settings.bootstrap_servers,
            "group.id": settings.group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "client.id": "paysim-postgres-consumer",
        }
    )
    summary = ConsumerSummary()
    consumer.subscribe([settings.topic])
    LOGGER.info("Listening for PaySim events on %s", settings.topic)
    try:
        while max_messages is None or summary.received < max_messages:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                if message.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(message.error())
            process_message(consumer, message, settings, postgres_settings, summary)
    except KeyboardInterrupt:
        LOGGER.info("Shutdown requested; closing consumer after current safe outcome")
    finally:
        consumer.close()

    LOGGER.info(
        "Consumer stopped: received=%s valid=%s invalid=%s transaction_inserts=%s "
        "event_inserts=%s duplicate_events=%s acknowledged=%s",
        summary.received,
        summary.valid,
        summary.invalid,
        summary.transactions_inserted,
        summary.events_inserted,
        summary.duplicate_events,
        summary.acknowledged,
    )
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the local consumer command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-messages", type=int)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    return parser


def main() -> int:
    """Run the consumer CLI with environment-based service configuration."""
    arguments = build_argument_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        consume_events(load_consumer_settings(), arguments.max_messages)
    except (KafkaException, psycopg.Error, ValueError) as error:
        LOGGER.error("Kafka consumer stopped with an error: %s", type(error).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
