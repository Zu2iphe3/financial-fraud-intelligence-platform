"""Configuration for the local Streamlit portfolio application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse


class DashboardConfigurationError(ValueError):
    """Raised when dashboard configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class DashboardSettings:
    """Local endpoints and presentation timeouts used by Streamlit."""

    prediction_api_base_url: str
    request_timeout_seconds: float
    health_cache_seconds: int


def load_dashboard_settings(
    environ: Mapping[str, str] | None = None,
) -> DashboardSettings:
    """Load local defaults while allowing environment-specific ports and paths."""

    environment = os.environ if environ is None else environ
    try:
        settings = DashboardSettings(
            prediction_api_base_url=environment.get(
                "FRAUD_API_BASE_URL", "http://127.0.0.1:8000"
            ).rstrip("/"),
            request_timeout_seconds=float(
                environment.get("DASHBOARD_REQUEST_TIMEOUT_SECONDS", "5")
            ),
            health_cache_seconds=int(environment.get("DASHBOARD_HEALTH_CACHE_SECONDS", "20")),
        )
    except (TypeError, ValueError) as error:
        raise DashboardConfigurationError("Dashboard configuration is invalid") from error

    parsed_api = urlparse(settings.prediction_api_base_url)
    if parsed_api.scheme not in {"http", "https"} or not parsed_api.hostname:
        raise DashboardConfigurationError("FRAUD_API_BASE_URL must be a valid HTTP URL")
    if parsed_api.username or parsed_api.password:
        raise DashboardConfigurationError("FRAUD_API_BASE_URL must not contain credentials")
    if not 0 < settings.request_timeout_seconds <= 30:
        raise DashboardConfigurationError(
            "DASHBOARD_REQUEST_TIMEOUT_SECONDS must be between 0 and 30"
        )
    if not 1 <= settings.health_cache_seconds <= 300:
        raise DashboardConfigurationError(
            "DASHBOARD_HEALTH_CACHE_SECONDS must be between 1 and 300"
        )
    return settings
