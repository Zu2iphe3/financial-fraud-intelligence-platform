"""Opt-in parity and real-model scoring validation for Stage 1 inference."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import pytest

from src.api.config import load_api_settings
from src.dashboard.config import load_dashboard_settings
from src.dashboard.prediction_client import PredictionClient
from src.dashboard.transaction_scoring import HistoricalTransactionScoringService
from src.data.postgres import load_postgres_settings
from src.features.contract import REPOSITORY_ROOT, load_feature_contract
from src.features.point_in_time import PointInTimeFeatureService
from src.models.scoring_artifact import load_scoring_bundle


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_POINT_IN_TIME_TESTS") != "1",
        reason="requires RUN_POINT_IN_TIME_TESTS=1, PostgreSQL, v2 Parquet, and FastAPI",
    ),
]

V2_ROOT = REPOSITORY_ROOT / "data/processed/paysim_features/paysim_features_v2"
SPECIAL_SAMPLE_IDS = {
    "24h_boundary": "c3f8e5e4-acfc-59d8-8a49-e14159f293dc",
    "168h_boundary": "172d66ff-6fe3-5b8a-8383-0ffb185786bc",
    "same_step_first": "18015d8f-11a5-5f08-a38e-fdc0d5b8ae8e",
    "same_step_second": "09e777e6-00b1-5b90-8fcc-e67ada3c0b68",
}


def _reviewed_samples(limit: int = 12) -> list[tuple[str, pd.Series]]:
    """Select diverse reviewed rows without loading the 6.3M-row matrix."""
    samples: list[tuple[str, pd.Series]] = []
    covered_types: set[str] = set()
    have_cold = False
    have_history = False
    for partition in ("train", "validation", "test"):
        for feature_path in sorted((V2_ROOT / partition / "features").glob("*.parquet")):
            metadata_path = V2_ROOT / partition / "metadata" / feature_path.name
            features = pd.read_parquet(feature_path)
            metadata = pd.read_parquet(metadata_path)
            for index, row in features.iterrows():
                transaction_type = next(
                    name.removeprefix("transaction_type_").upper()
                    for name in features.columns
                    if name.startswith("transaction_type_") and int(row[name]) == 1
                )
                cold = int(row["dest_history_available"]) == 0
                useful = (
                    transaction_type not in covered_types
                    or (cold and not have_cold)
                    or (not cold and not have_history)
                )
                if not useful:
                    continue
                samples.append((str(metadata.iloc[index]["transaction_id"]), row))
                covered_types.add(transaction_type)
                have_cold |= cold
                have_history |= not cold
                if len(covered_types) == 5 and have_cold and have_history:
                    return samples[:limit]
    return samples[:limit]


def _late_history_sample() -> str:
    """Use a late transaction so the plan must avoid scanning almost all source rows."""
    for feature_path in sorted((V2_ROOT / "test" / "features").glob("*.parquet"), reverse=True):
        metadata_path = V2_ROOT / "test" / "metadata" / feature_path.name
        features = pd.read_parquet(feature_path, columns=["dest_history_available"])
        metadata = pd.read_parquet(metadata_path, columns=["transaction_id", "step"])
        candidates = features["dest_history_available"].eq(1) & metadata["step"].ge(700)
        if candidates.any():
            return str(metadata.loc[candidates.idxmax(), "transaction_id"])
    raise AssertionError("A late reviewed transaction with destination history is required")


def _reviewed_row(transaction_id: str) -> pd.Series:
    """Locate one stable PaySim identity in the reviewed, partitioned v2 output."""
    for partition in ("train", "validation", "test"):
        for metadata_path in sorted((V2_ROOT / partition / "metadata").glob("*.parquet")):
            metadata = pd.read_parquet(metadata_path, columns=["transaction_id"])
            matches = metadata["transaction_id"].astype(str).eq(transaction_id)
            if matches.any():
                feature_path = V2_ROOT / partition / "features" / metadata_path.name
                return pd.read_parquet(feature_path).loc[matches.idxmax()]
    raise AssertionError(f"Reviewed transaction {transaction_id} is missing")


def _assert_feature_parity(actual: pd.Series, expected: pd.Series) -> None:
    assert tuple(actual.index) == tuple(expected.index)
    for feature in expected.index:
        if str(expected[feature].dtype).startswith("int"):
            assert int(actual[feature]) == int(expected[feature])
        else:
            # Both pipelines intentionally store float32; one ULP-scale tolerance
            # protects against platform-specific aggregation order only.
            np.testing.assert_allclose(actual[feature], expected[feature], rtol=1e-6, atol=1e-6)


def test_reconstructed_features_match_reviewed_v2_and_score_real_model() -> None:
    assert V2_ROOT.is_dir(), "Reviewed v2 Parquet dataset is required"
    settings = load_postgres_settings()
    feature_service = PointInTimeFeatureService(settings)
    review_transactions = feature_service.list_review_transactions(5)
    assert len(review_transactions) == 5
    assert all(transaction.transaction_id for transaction in review_transactions)
    samples = _reviewed_samples()
    assert len(samples) >= 5

    for transaction_id, expected in samples:
        actual = feature_service.reconstruct(transaction_id).features.iloc[0]
        _assert_feature_parity(actual, expected)

    for transaction_id in SPECIAL_SAMPLE_IDS.values():
        actual = feature_service.reconstruct(transaction_id).features.iloc[0]
        _assert_feature_parity(actual, _reviewed_row(transaction_id))

    # Both transactions share destination and step 1, so neither can observe the
    # other despite their different source-row positions within the CSV.
    for key in ("same_step_first", "same_step_second"):
        same_step = feature_service.reconstruct(SPECIAL_SAMPLE_IDS[key]).features.iloc[0]
        assert same_step["dest_history_available"] == 0
        assert same_step["dest_tx_count_log1p_24h"] == 0

    scoring = HistoricalTransactionScoringService(
        feature_service,
        PredictionClient(
            load_dashboard_settings().prediction_api_base_url,
            load_dashboard_settings().request_timeout_seconds,
        ),
    )
    result = scoring.score(samples[0][0])
    assert 0 <= result.prediction.fraud_probability <= 1
    assert result.prediction.risk_classification in {"low", "moderate", "high"}
    direct_probability = load_scoring_bundle(
        load_api_settings().model_artifact_path
    ).predict_probabilities(result.prepared.features)[0]
    assert result.prediction.fraud_probability == pytest.approx(
        direct_probability, abs=1e-7
    )


def test_history_lookup_query_uses_the_reviewed_index() -> None:
    settings = load_postgres_settings()
    sample_id = _late_history_sample()
    with psycopg.connect(**settings.connection_kwargs()) as connection:
        selected = connection.execute(
            """
            SELECT source_dataset_id, name_dest, step
            FROM analytics_marts.fct_transactions
            WHERE transaction_id = %s
            """,
            (sample_id,),
        ).fetchone()
        plan = connection.execute(
            """
            EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
            SELECT source_row_number, step, transaction_type, amount, name_orig
            FROM raw.transactions
            WHERE source_dataset_id = %s AND name_dest = %s AND step < %s
            ORDER BY step, source_row_number
            """,
            selected,
        ).fetchall()

    plan_text = "\n".join(row[0] for row in plan)
    assert "transactions_dataset_destination_step_row_idx" in plan_text
