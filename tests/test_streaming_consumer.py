from __future__ import annotations

import logging

import psycopg
import pytest

from src.data.paysim_ingestion import parse_paysim_row
from src.data.postgres import PostgresSettings
from src.streaming.consumer import (
    ConsumerSettings,
    ConsumerSummary,
    KafkaPosition,
    PersistenceOutcome,
    persist_event_with_retry,
    process_message,
)
from src.streaming.event_contract import build_event_payload, parse_event_payload


def sample_transaction():
    return parse_paysim_row(
        {
            "step": "12",
            "type": "TRANSFER",
            "amount": "100.10",
            "nameOrig": "C100",
            "oldbalanceOrg": "500.25",
            "newbalanceOrig": "400.15",
            "nameDest": "C200",
            "oldbalanceDest": "25.00",
            "newbalanceDest": "125.10",
            "isFraud": "1",
            "isFlaggedFraud": "0",
        },
        7,
        "a" * 64,
        "paysim.csv",
    )


class FakeMessage:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

    def topic(self):
        return "events"

    def partition(self):
        return 0

    def offset(self):
        return 12


class FakeConsumer:
    def __init__(self, actions):
        self.actions = actions

    def commit(self, message, asynchronous):
        assert asynchronous is False
        self.actions.append(("commit", message.offset()))


SETTINGS = ConsumerSettings("broker:9092", "events", "group", 3, 0)
POSTGRES = PostgresSettings("database", "user", "password", "host", 5432)


def test_valid_event_is_acknowledged_only_after_persistence():
    actions = []
    consumer = FakeConsumer(actions)
    summary = ConsumerSummary()

    def persistence(event, position, _postgres, _attempts, _backoff):
        actions.append(("persist", event.event_id, position.offset))
        return PersistenceOutcome(True, True)

    process_message(
        consumer,
        FakeMessage(build_event_payload(sample_transaction())),
        SETTINGS,
        POSTGRES,
        summary,
        persistence=persistence,
    )

    assert [action[0] for action in actions] == ["persist", "commit"]
    assert summary.received == summary.valid == summary.acknowledged == 1
    assert summary.transactions_inserted == summary.events_inserted == 1


def test_invalid_event_is_diagnosed_without_logging_payload(caplog):
    actions = []
    summary = ConsumerSummary()
    secret_payload = b'{"customer_secret":"never-log-this"}'

    with caplog.at_level(logging.ERROR):
        process_message(FakeConsumer(actions), FakeMessage(secret_payload), SETTINGS, POSTGRES, summary)

    assert actions == [("commit", 12)]
    assert summary.invalid == summary.acknowledged == 1
    assert "validation=" in caplog.text
    assert "never-log-this" not in caplog.text


def test_database_failure_does_not_acknowledge_event():
    actions = []

    def unavailable(*_arguments):
        raise psycopg.OperationalError("database unavailable")

    with pytest.raises(psycopg.OperationalError):
        process_message(
            FakeConsumer(actions),
            FakeMessage(build_event_payload(sample_transaction())),
            SETTINGS,
            POSTGRES,
            ConsumerSummary(),
            persistence=unavailable,
        )

    assert actions == []


def test_database_retry_policy_uses_all_attempts_before_failure():
    event = parse_event_payload(build_event_payload(sample_transaction()))
    attempts = []
    sleeps = []

    def unavailable(**_kwargs):
        attempts.append("connect")
        raise psycopg.OperationalError("database unavailable")

    with pytest.raises(psycopg.OperationalError):
        persist_event_with_retry(
            event,
            KafkaPosition("events", 0, 1),
            POSTGRES,
            attempts=3,
            backoff_seconds=2,
            connect=unavailable,
            sleeper=sleeps.append,
        )

    assert attempts == ["connect", "connect", "connect"]
    assert sleeps == [2, 4]
