"""Independent, lightweight readiness checks for dashboard dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import httpx
import psycopg

from src.dashboard.config import DashboardSettings
from src.data.postgres import load_postgres_settings
from src.rag.config import load_rag_settings


HealthState = Literal["available", "unavailable", "setup_required", "checking"]


@dataclass(frozen=True)
class ServiceHealth:
    key: str
    label: str
    state: HealthState
    message: str

    @property
    def available(self) -> bool:
        return self.state == "available"


def pending_service_health() -> tuple[ServiceHealth, ...]:
    """Describe honest startup state before the real dependency checks finish."""

    return tuple(
        ServiceHealth(key, label, "checking", "Checking availability")
        for key, label in (
            ("postgres", "Analytics"),
            ("fastapi", "Risk scoring"),
            ("rag", "Policy search"),
            ("ollama", "Ollama"),
        )
    )


def check_postgres_analytics(connect: Callable[..., Any] = psycopg.connect) -> ServiceHealth:
    """Confirm dashboard marts are queryable without reading raw transaction data."""

    try:
        settings = load_postgres_settings()
        with connect(**settings.connection_kwargs(), connect_timeout=3) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    select
                        to_regclass('analytics_marts.mart_fraud_by_day') is not null,
                        to_regclass('analytics_marts.mart_fraud_by_transaction_type') is not null
                    """
                )
                marts_exist = all(cursor.fetchone())
                if marts_exist:
                    cursor.execute(
                        """
                        select
                            exists(select 1 from analytics_marts.mart_fraud_by_day limit 1),
                            exists(
                                select 1
                                from analytics_marts.mart_fraud_by_transaction_type
                                limit 1
                            )
                        """
                    )
                    available = all(cursor.fetchone())
                else:
                    available = False
    except Exception:
        return ServiceHealth(
            "postgres", "Analytics", "unavailable", "Historical data unavailable"
        )
    if not available:
        return ServiceHealth(
            "postgres", "Analytics", "setup_required", "Prepare the analytics data"
        )
    return ServiceHealth("postgres", "Analytics", "available", "Historical data ready")


def check_prediction_api(
    settings: DashboardSettings, get: Callable[..., Any] = httpx.get,
) -> ServiceHealth:
    try:
        response = get(
            f"{settings.prediction_api_base_url}/health/ready",
            timeout=settings.request_timeout_seconds,
        )
        payload = response.json()
        ready = response.status_code == 200 and payload.get("prediction_available") is True
    except Exception:
        ready = False
    if not ready:
        return ServiceHealth(
            "fastapi", "Risk scoring", "unavailable", "Prediction service unavailable"
        )
    return ServiceHealth("fastapi", "Risk scoring", "available", "Prediction service ready")


def check_policy_search(
    service_factory: Callable[[], Any] | None = None,
) -> ServiceHealth:
    """Run one real semantic query so health matches the user-facing search path."""

    try:
        if service_factory is None:
            # SentenceTransformer and Chroma are intentionally imported only when
            # this real check runs. Importing them before st.navigation is what made
            # Streamlit expose its legacy filesystem page list during cold startup.
            from src.dashboard.policy_service import get_dashboard_policy_service

            service_factory = get_dashboard_policy_service
        settings = load_rag_settings()
        if not settings.vector_store_path.is_dir():
            return ServiceHealth(
                "rag", "Policy search", "setup_required", "Prepare the policy library"
            )
        service = service_factory()
        evidence_count = service.check_search_readiness()
    except Exception:
        return ServiceHealth(
            "rag", "Policy search", "unavailable", "Semantic search unavailable"
        )
    return ServiceHealth(
        "rag",
        "Policy search",
        "available",
        f"Semantic retrieval ready · {evidence_count} evidence matches",
    )


def check_ollama(get: Callable[..., Any] = httpx.get) -> ServiceHealth:
    try:
        settings = load_rag_settings()
        response = get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        response.raise_for_status()
        names = {str(item.get("name", "")) for item in response.json().get("models", [])}
        configured = settings.ollama_chat_model
        ready = configured in names or f"{configured}:latest" in names
    except Exception:
        ready = False
    if not ready:
        return ServiceHealth(
            "ollama", "Ollama", "unavailable", "Configured local model unavailable"
        )
    return ServiceHealth("ollama", "Ollama", "available", "Local generation ready")


def collect_service_health(settings: DashboardSettings) -> tuple[ServiceHealth, ...]:
    """Run checks independently so one failed dependency cannot hide another."""

    return (
        check_postgres_analytics(),
        check_prediction_api(settings),
        check_policy_search(),
        check_ollama(),
    )
