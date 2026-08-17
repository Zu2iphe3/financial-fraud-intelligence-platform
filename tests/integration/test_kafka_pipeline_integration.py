from __future__ import annotations

import csv
import os
import uuid

import psycopg
import pytest
from confluent_kafka.admin import AdminClient, NewTopic

from src.data.postgres import load_postgres_settings
from src.data.transaction_identity import calculate_dataset_sha256
from src.streaming.consumer import ConsumerSettings, consume_events
from src.streaming.producer import ProducerSettings, replay_paysim


pytestmark = [pytest.mark.integration, pytest.mark.kafka]


def _kafka_tests_enabled() -> bool:
    return os.getenv("RUN_KAFKA_TESTS") == "1" and os.getenv("RUN_POSTGRES_TESTS") == "1"


def _row(index: int) -> dict[str, str]:
    return {
        "step": str(index),
        "type": "CASH_OUT",
        "amount": f"{index}.25",
        "nameOrig": f"C_E2E_{index}",
        "oldbalanceOrg": "500.00",
        "newbalanceOrig": "400.00",
        "nameDest": f"C_E2E_DEST_{index}",
        "oldbalanceDest": "10.00",
        "newbalanceDest": "110.00",
        "isFraud": "0",
        "isFlaggedFraud": "0",
    }


def _write_source(path):
    with path.open("w", encoding="utf-8", newline="") as source:
        writer = csv.DictWriter(source, fieldnames=tuple(_row(1)))
        writer.writeheader()
        writer.writerows(_row(index) for index in range(1, 4))


def test_known_replay_count_reconciles_across_kafka_and_postgres(tmp_path):
    if not _kafka_tests_enabled():
        pytest.skip("Set RUN_KAFKA_TESTS=1 and RUN_POSTGRES_TESTS=1 for Kafka integration")

    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    suffix = uuid.uuid4().hex[:10]
    topic = f"paysim-integration-{suffix}"
    group_id = f"paysim-integration-group-{suffix}"
    source = tmp_path / "paysim-kafka-integration.csv"
    _write_source(source)
    dataset_id = calculate_dataset_sha256(source)
    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])[topic].result(30)

    postgres_settings = load_postgres_settings()
    try:
        producer_settings = ProducerSettings(bootstrap_servers, topic, 1, 3, 1_000)
        consumer_settings = ConsumerSettings(bootstrap_servers, topic, group_id, 3, 0)

        first_production = replay_paysim(producer_settings, source)
        first_consumption = consume_events(consumer_settings, max_messages=3)
        second_production = replay_paysim(producer_settings, source)
        second_consumption = consume_events(consumer_settings, max_messages=3)

        with psycopg.connect(**postgres_settings.connection_kwargs()) as connection:
            transaction_count = connection.execute(
                "SELECT COUNT(*) FROM raw.transactions WHERE source_dataset_id = %s",
                (dataset_id,),
            ).fetchone()[0]
            ingestion_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM raw.transaction_ingestions ingestion
                JOIN raw.transactions txn USING (transaction_id)
                WHERE txn.source_dataset_id = %s
                """,
                (dataset_id,),
            ).fetchone()[0]

        assert first_production.acknowledged == second_production.acknowledged == 3
        assert first_consumption.received == first_consumption.acknowledged == 3
        assert first_consumption.events_inserted == 3
        assert second_consumption.received == second_consumption.acknowledged == 3
        assert second_consumption.duplicate_events == 3
        assert transaction_count == ingestion_count == 3
    finally:
        with psycopg.connect(**postgres_settings.connection_kwargs(), autocommit=True) as connection:
            connection.execute(
                "DELETE FROM raw.transaction_ingestions WHERE transaction_id IN "
                "(SELECT transaction_id FROM raw.transactions WHERE source_dataset_id = %s)",
                (dataset_id,),
            )
            connection.execute(
                "DELETE FROM raw.transactions WHERE source_dataset_id = %s", (dataset_id,)
            )
        admin.delete_topics([topic])[topic].result(30)
