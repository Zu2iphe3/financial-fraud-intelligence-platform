"""Opt-in validation against the real analytics marts and prediction API."""

from __future__ import annotations

import os

import pytest

from src.dashboard.analytics import AnalyticsFilters, AnalyticsRepository
from src.dashboard.config import load_dashboard_settings
from src.dashboard.prediction_client import PredictionClient
from src.dashboard.transaction_scoring import HistoricalTransactionScoringService
from src.data.postgres import load_postgres_settings
from src.features.point_in_time import PointInTimeFeatureService


pytestmark = [
    pytest.mark.dashboard,
    pytest.mark.skipif(
        os.getenv("RUN_DASHBOARD_TESTS") != "1",
        reason="requires RUN_DASHBOARD_TESTS=1, PostgreSQL marts, and FastAPI",
    ),
]


def test_real_dashboard_analytics_and_prediction_dependencies() -> None:
    dashboard_settings = load_dashboard_settings()
    repository = AnalyticsRepository(load_postgres_settings())
    min_day, max_day, transaction_types = repository.load_filter_options()

    snapshot = repository.load_snapshot(
        AnalyticsFilters(min_day, max_day, transaction_types)
    )
    response = HistoricalTransactionScoringService(
        PointInTimeFeatureService(load_postgres_settings()),
        PredictionClient(
            dashboard_settings.prediction_api_base_url,
            dashboard_settings.request_timeout_seconds,
        ),
    ).score("bc7ae75a-3f04-5ad3-927b-741185385e87").prediction

    assert snapshot.metrics.transaction_count == 6_362_620
    assert snapshot.metrics.fraud_count == 8_213
    assert response.fraud_probability >= 0
    assert response.risk_classification in {"low", "moderate", "high"}
