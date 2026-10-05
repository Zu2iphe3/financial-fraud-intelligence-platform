"""Structural and unavailable-service smoke tests for the Streamlit application."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from src.api.contracts import PredictionRequest
from src.dashboard import health
from src.dashboard.health import ServiceHealth


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_navigation_declares_three_distinct_professional_experiences() -> None:
    source = (REPOSITORY_ROOT / "src" / "dashboard" / "app.py").read_text(encoding="utf-8")

    assert "Overview & Analytics" in source
    assert "Fraud Risk Scoring" in source
    assert "Policy Assistant" in source
    assert "st.navigation" in source


def test_navigation_and_loading_state_render_before_health_initialization() -> None:
    source = (REPOSITORY_ROOT / "src" / "dashboard" / "app.py").read_text(
        encoding="utf-8"
    )

    assert source.index("navigation = st.navigation") < source.index(
        "from src.dashboard.health import"
    )
    assert 'position="hidden"' in source
    assert "st.sidebar.page_link" in source
    assert source.index("startup_notice.info") < source.index(
        "from src.dashboard.health import"
    )
    assert source.index("navigation.run()") < source.index(
        "resolved_statuses = _service_health()"
    )


def test_legacy_navigation_is_hidden_during_cold_start() -> None:
    config = (REPOSITORY_ROOT / ".streamlit" / "config.toml").read_text(
        encoding="utf-8"
    )

    assert "showSidebarNavigation = false" in config


def test_expensive_page_dependencies_load_after_page_shell() -> None:
    overview = (
        REPOSITORY_ROOT / "src" / "dashboard" / "pages" / "overview.py"
    ).read_text(encoding="utf-8")
    policy = (
        REPOSITORY_ROOT / "src" / "dashboard" / "pages" / "policy_assistant.py"
    ).read_text(encoding="utf-8")

    assert overview.index('"Overview & Analytics"') < overview.index(
        "from src.dashboard.analytics import AnalyticsUnavailableError"
    )
    assert policy.index('"Policy Assistant"') < policy.index(
        "from src.dashboard.policy_service import ("
    )


def test_user_facing_dashboard_copy_omits_internal_ticket_labels() -> None:
    presentation_files = [
        REPOSITORY_ROOT / "src" / "dashboard" / "app.py",
        REPOSITORY_ROOT / "src" / "dashboard" / "ui.py",
        *(REPOSITORY_ROOT / "src" / "dashboard" / "pages").glob("*.py"),
    ]

    presentation = "\n".join(
        path.read_text(encoding="utf-8") for path in presentation_files
    )

    assert "PH9-02" not in presentation
    assert "PH10-01" not in presentation


def test_overview_distinguishes_labelled_analytics_from_model_predictions() -> None:
    source = (
        REPOSITORY_ROOT / "src" / "dashboard" / "pages" / "overview.py"
    ).read_text(encoding="utf-8")

    assert "Labelled fraud cases" in source
    assert "Labelled fraud rate" in source
    assert "separate from ML fraud-risk predictions" in source
    assert "prediction_client" not in source


def test_prediction_contract_and_form_exclude_prohibited_inputs() -> None:
    scoring_source = (
        REPOSITORY_ROOT / "src" / "dashboard" / "pages" / "fraud_scoring.py"
    ).read_text(encoding="utf-8")

    prohibited_model_inputs = {
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    }
    assert set(PredictionRequest.model_fields).isdisjoint(prohibited_model_inputs)
    assert all(field not in scoring_source for field in prohibited_model_inputs)
    assert "number_input" not in scoring_source
    assert "checkbox" not in scoring_source


def test_dashboard_renders_when_all_dependencies_are_unavailable(monkeypatch, tmp_path) -> None:
    # Keep the UI smoke test deterministic. Individual health-check tests below this
    # layer exercise real connection outcomes without making Streamlit's test runner
    # wait on several deliberately unreachable Windows sockets.
    monkeypatch.setattr(
        health,
        "collect_service_health",
        lambda _settings: (
            ServiceHealth("postgres", "Analytics", "unavailable", "Unavailable"),
            ServiceHealth("fastapi", "Risk scoring", "unavailable", "Unavailable"),
            ServiceHealth("rag", "Policy search", "setup_required", "Setup required"),
            ServiceHealth("ollama", "Ollama", "unavailable", "Unavailable"),
        ),
    )
    monkeypatch.setenv("POSTGRES_DB", "unavailable")
    monkeypatch.setenv("POSTGRES_USER", "unavailable")
    monkeypatch.setenv("POSTGRES_PASSWORD", "local-test-only")
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PORT", "1")
    monkeypatch.setenv("FRAUD_API_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("RAG_CHROMA_PATH", str(tmp_path / "missing-index"))
    monkeypatch.setenv("DASHBOARD_REQUEST_TIMEOUT_SECONDS", "0.2")

    app = AppTest.from_file(
        str(REPOSITORY_ROOT / "src" / "dashboard" / "app.py"),
        # A deliberately unreachable PostgreSQL socket may consume both its page
        # and health-check timeouts before the controlled unavailable state renders.
        default_timeout=60,
    ).run()

    assert not app.exception
    assert app.title[0].value == "Overview & Analytics"
    overview_text = " ".join(item.value for item in app.markdown)
    assert "Services" in overview_text
    assert "Analytics" in overview_text
    assert "Risk scoring" in overview_text
    assert "Policy search" in overview_text
    assert "Ollama" in overview_text

    scoring = app.switch_page("pages/fraud_scoring.py").run()
    assert not scoring.exception
    assert scoring.title[0].value == "Fraud Risk Scoring"
    assert scoring.button(key="FormSubmitter:fraud_scoring_form-Analyse risk")

    policy = scoring.switch_page("pages/policy_assistant.py").run()
    assert not policy.exception
    assert policy.title[0].value == "Policy Assistant"
    assert policy.text_area(key="policy_question")


def test_page_navigation_reuses_cached_service_health(monkeypatch) -> None:
    calls = 0

    def collect_once(_settings):
        nonlocal calls
        calls += 1
        return (
            ServiceHealth("postgres", "Analytics", "available", "Ready"),
            ServiceHealth("fastapi", "Risk scoring", "available", "Ready"),
            ServiceHealth("rag", "Policy search", "available", "Ready"),
            ServiceHealth("ollama", "Ollama", "available", "Ready"),
        )

    st.cache_data.clear()
    monkeypatch.setattr(health, "collect_service_health", collect_once)
    app = AppTest.from_file(
        str(REPOSITORY_ROOT / "src" / "dashboard" / "app.py"),
        # A fresh policy-page import loads the local RAG modules even though
        # cached health results prevent repeated service initialization.
        default_timeout=60,
    ).run()
    scoring = app.switch_page("pages/fraud_scoring.py").run()
    policy = scoring.switch_page("pages/policy_assistant.py").run()

    assert not policy.exception
    assert calls == 1
    st.cache_data.clear()
