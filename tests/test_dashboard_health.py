"""Tests for independent dashboard dependency-health checks."""

from __future__ import annotations

from src.dashboard import health
from src.dashboard.config import load_dashboard_settings
from src.data.postgres import PostgresSettings
from src.rag.config import load_rag_settings


class ContextCursor:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, *_):
        return None

    def fetchone(self):
        return (True, True)


class ContextConnection:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return ContextCursor()


class FakeResponse:
    def __init__(self, status_code=200, payload=None) -> None:
        self.status_code = status_code
        self.payload = payload or {}

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("unavailable")


def test_postgres_health_reports_marts_without_exposing_credentials(monkeypatch) -> None:
    monkeypatch.setattr(
        health,
        "load_postgres_settings",
        lambda: PostgresSettings("fraud", "user", "secret", "localhost", 5432),
    )

    result = health.check_postgres_analytics(connect=lambda **_: ContextConnection())

    assert result.available
    assert "secret" not in result.message


def test_prediction_health_reports_ready_and_unavailable_independently() -> None:
    settings = load_dashboard_settings(environ={})
    ready = health.check_prediction_api(
        settings,
        get=lambda *_, **__: FakeResponse(200, {"prediction_available": True}),
    )
    unavailable = health.check_prediction_api(
        settings,
        get=lambda *_, **__: (_ for _ in ()).throw(OSError("private")),
    )

    assert ready.available
    assert unavailable.state == "unavailable"
    assert "private" not in unavailable.message


def test_policy_search_health_runs_a_real_semantic_readiness_query(
    monkeypatch, tmp_path
) -> None:
    index_path = tmp_path / "policy-index"
    index_path.mkdir()
    settings = load_rag_settings(environ={"RAG_CHROMA_PATH": str(index_path)})
    monkeypatch.setattr(health, "load_rag_settings", lambda: settings)

    class ReadyService:
        @staticmethod
        def check_search_readiness():
            return 4

    result = health.check_policy_search(service_factory=ReadyService)

    assert result.available
    assert result.message == "Semantic retrieval ready · 4 evidence matches"


def test_policy_search_health_reports_embedding_initialization_failure(
    monkeypatch, tmp_path
) -> None:
    index_path = tmp_path / "policy-index"
    index_path.mkdir()
    settings = load_rag_settings(environ={"RAG_CHROMA_PATH": str(index_path)})
    monkeypatch.setattr(health, "load_rag_settings", lambda: settings)

    result = health.check_policy_search(
        service_factory=lambda: (_ for _ in ()).throw(
            RuntimeError("private embedding error")
        )
    )

    assert result.state == "unavailable"
    assert result.message == "Semantic search unavailable"
    assert "private" not in result.message


def test_rag_health_does_not_create_a_missing_index(monkeypatch, tmp_path) -> None:
    missing_path = tmp_path / "missing-index"
    settings = load_rag_settings(
        environ={"RAG_CHROMA_PATH": str(missing_path)}
    )
    monkeypatch.setattr(health, "load_rag_settings", lambda: settings)

    result = health.check_policy_search(
        service_factory=lambda: (_ for _ in ()).throw(AssertionError("must not open"))
    )

    assert result.state == "setup_required"
    assert not missing_path.exists()


def test_ollama_health_requires_configured_model(monkeypatch) -> None:
    settings = load_rag_settings(environ={})
    monkeypatch.setattr(health, "load_rag_settings", lambda: settings)

    ready = health.check_ollama(
        get=lambda *_, **__: FakeResponse(200, {"models": [{"name": "llama3.2:3b"}]})
    )
    unavailable = health.check_ollama(
        get=lambda *_, **__: FakeResponse(200, {"models": []})
    )

    assert ready.available
    assert unavailable.state == "unavailable"


def test_pending_health_never_claims_availability() -> None:
    pending = health.pending_service_health()

    assert [item.label for item in pending] == [
        "Analytics",
        "Risk scoring",
        "Policy search",
        "Ollama",
    ]
    assert all(item.state == "checking" for item in pending)
    assert not any(item.available for item in pending)
