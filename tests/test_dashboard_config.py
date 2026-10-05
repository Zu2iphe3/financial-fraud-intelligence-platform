"""Tests for safe local dashboard configuration."""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from src.dashboard.config import DashboardConfigurationError, load_dashboard_settings


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_disables_dynamic_torch_module_watching() -> None:
    config = tomllib.loads(
        (REPOSITORY_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    )

    assert config["server"]["fileWatcherType"] == "none"


def test_dashboard_defaults_are_local() -> None:
    settings = load_dashboard_settings(environ={})

    assert settings.prediction_api_base_url == "http://127.0.0.1:8000"
    assert settings.request_timeout_seconds == 5


@pytest.mark.parametrize(
    "environment, message",
    [
        ({"FRAUD_API_BASE_URL": "not-a-url"}, "valid HTTP URL"),
        ({"FRAUD_API_BASE_URL": "http://user:secret@localhost:8000"}, "credentials"),
        ({"DASHBOARD_REQUEST_TIMEOUT_SECONDS": "0"}, "between 0 and 30"),
        ({"DASHBOARD_HEALTH_CACHE_SECONDS": "0"}, "between 1 and 300"),
    ],
)
def test_dashboard_rejects_invalid_configuration(environment, message) -> None:
    with pytest.raises(DashboardConfigurationError, match=message):
        load_dashboard_settings(environ=environment)
