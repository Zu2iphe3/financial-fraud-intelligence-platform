"""PostgreSQL integration test for chunked, versioned feature generation."""

from __future__ import annotations

import os
from dataclasses import replace
from decimal import Decimal
from uuid import uuid4

import pandas as pd
import psycopg
import pytest
from psycopg import sql

from src.data.postgres import load_postgres_settings
from src.features.build_paysim_dataset import build_modelling_dataset
from src.features.contract import load_feature_contract


pytestmark = pytest.mark.integration


@pytest.fixture
def connection():
    if os.getenv("RUN_POSTGRES_TESTS") != "1":
        pytest.skip("Set RUN_POSTGRES_TESTS=1 to run PostgreSQL integration tests")
    settings = load_postgres_settings()
    with psycopg.connect(**settings.connection_kwargs()) as database_connection:
        yield database_connection


def test_chunked_dataset_generation_reconciles_partitions_and_outputs(connection, tmp_path):
    schema_name = f"feature_test_{uuid4().hex}"
    relation = sql.Identifier(schema_name)
    connection.execute(sql.SQL("CREATE SCHEMA {}").format(relation))
    connection.execute(
        sql.SQL(
            """
            CREATE TABLE {}.transactions (
                transaction_id TEXT NOT NULL,
                source_dataset_id TEXT NOT NULL,
                source_row_number BIGINT NOT NULL,
                step INTEGER NOT NULL,
                simulation_day INTEGER NOT NULL,
                transaction_type TEXT NOT NULL,
                amount NUMERIC(20, 2) NOT NULL,
                is_fraud SMALLINT NOT NULL
            )
            """
        ).format(relation)
    )
    rows = []
    for row_number, step in enumerate((1, 432, 433, 576, 577, 720, 721, 743), start=1):
        rows.append(
            (
                f"transaction-{row_number}",
                "a" * 64,
                row_number,
                step,
                ((step - 1) // 24) + 1,
                "TRANSFER" if row_number % 2 else "PAYMENT",
                Decimal(f"{row_number * 100}.00"),
                1 if step >= 721 else 0,
            )
        )
    with connection.cursor() as cursor:
        cursor.executemany(
            sql.SQL("INSERT INTO {}.transactions VALUES (%s, %s, %s, %s, %s, %s, %s, %s)").format(
                relation
            ),
            rows,
        )
    connection.commit()

    try:
        contract = replace(
            load_feature_contract(), source_relation=f"{schema_name}.transactions"
        )
        output = tmp_path / "paysim_features_v1"

        manifest = build_modelling_dataset(
            connection,
            contract,
            output,
            chunk_size=3,
            validate_expected_facts=False,
        )

        assert manifest["contract_version"] == "paysim_features_v1"
        assert [manifest["splits"][name]["rows"] for name in ("train", "validation", "test")] == [
            2,
            2,
            4,
        ]
        assert manifest["splits"]["test"]["final_day_rows"] == 2
        assert manifest["splits"]["test"]["final_day_fraud_rows"] == 2
        train_features = pd.concat(
            pd.read_parquet(path)
            for path in sorted((output / "train" / "features").glob("*.parquet"))
        )
        assert tuple(train_features.columns) == contract.feature_names
        assert not set(contract.prohibited_predictive_sources).intersection(
            train_features.columns
        )
        assert (output / "manifest.json").is_file()
    finally:
        connection.rollback()
        connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(relation))
        connection.commit()
