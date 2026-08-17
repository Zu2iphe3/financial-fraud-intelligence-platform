from __future__ import annotations

import csv

from src.streaming.event_contract import parse_event_payload
from src.streaming.producer import (
    ProducerSettings,
    ReplayPacer,
    iter_replay_transactions,
    replay_paysim,
)
from src.data.transaction_identity import calculate_dataset_sha256


def _row(index: int) -> dict[str, str]:
    return {
        "step": str(index),
        "type": "PAYMENT",
        "amount": f"{index}.10",
        "nameOrig": f"C{index}",
        "oldbalanceOrg": "100.10",
        "newbalanceOrig": "99.00",
        "nameDest": f"M{index}",
        "oldbalanceDest": "0.00",
        "newbalanceDest": f"{index}.10",
        "isFraud": "0",
        "isFlaggedFraud": "0",
    }


def _csv(path, row_count: int = 4):
    columns = tuple(_row(1))
    with path.open("w", encoding="utf-8", newline="") as source:
        writer = csv.DictWriter(source, fieldnames=columns)
        writer.writeheader()
        writer.writerows(_row(index) for index in range(1, row_count + 1))
    return path


class FakeProducer:
    def __init__(self, _configuration):
        self.messages = []

    def produce(self, topic, key, value, on_delivery):
        self.messages.append((topic, key, value))
        on_delivery(None, object())

    def poll(self, _timeout):
        return 0

    def flush(self, _timeout):
        return 0


def test_replay_window_preserves_original_source_row_numbers(tmp_path):
    source = _csv(tmp_path / "paysim.csv")
    transactions = list(
        iter_replay_transactions(source, calculate_dataset_sha256(source), start_row=2, limit=2)
    )

    assert [item.source_row_number for item in transactions] == [2, 3]
    assert [str(item.amount) for item in transactions] == ["2.10", "3.10"]


def test_producer_publishes_exact_requested_count_of_valid_events(tmp_path):
    source = _csv(tmp_path / "paysim.csv")
    created = []

    def factory(configuration):
        producer = FakeProducer(configuration)
        created.append(producer)
        return producer

    summary = replay_paysim(
        ProducerSettings("broker:9092", "events", start_row=2, limit=2, rate_per_second=1_000),
        source,
        producer_factory=factory,
    )

    assert summary.requested == summary.attempted == summary.acknowledged == 2
    assert summary.failed == 0
    assert len(created[0].messages) == 2
    assert all(parse_event_payload(message[2]) for message in created[0].messages)


def test_replay_pacer_uses_configurable_interval():
    current = [10.0]
    sleeps = []

    def clock():
        return current[0]

    def sleeper(delay):
        sleeps.append(delay)
        current[0] += delay

    pacer = ReplayPacer(rate_per_second=4, clock=clock, sleeper=sleeper)
    pacer.wait()
    pacer.wait()
    pacer.wait()

    assert sleeps == [0.25, 0.25]
