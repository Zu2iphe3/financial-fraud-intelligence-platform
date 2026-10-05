"""Programmatic multi-page entry point for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.dashboard.ui import apply_dashboard_styles


PAGE_ROOT = Path(__file__).resolve().parent / "pages"
PAGE_DEFINITIONS = (
    ("overview.py", "Overview & Analytics", ":material/monitoring:"),
    ("fraud_scoring.py", "Fraud Risk Scoring", ":material/shield:"),
    ("policy_assistant.py", "Policy Assistant", ":material/menu_book:"),
)

st.set_page_config(
    page_title="Financial Fraud Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_dashboard_styles()

# Register routes without Streamlit's automatic sidebar. Explicit page links below
# render consistently while slow local dependencies are still being checked.
navigation = st.navigation(
    [
        st.Page(PAGE_ROOT / filename, title=title, icon=icon, default=index == 0)
        for index, (filename, title, icon) in enumerate(PAGE_DEFINITIONS)
    ],
    position="hidden",
)

for filename, title, icon in PAGE_DEFINITIONS:
    st.sidebar.page_link(PAGE_ROOT / filename, label=title, icon=icon)

st.sidebar.title("Fraud Intelligence")

initializing = "dashboard_service_health" not in st.session_state
startup_notice = st.empty()
if initializing:
    startup_notice.info(
        "Preparing dashboard connections. The Overview is loading while service "
        "availability is verified."
    )

# These imports deliberately follow st.navigation and the startup notice. The policy
# service itself remains lazy and cached, so this does not duplicate model loading.
from src.dashboard.config import load_dashboard_settings
from src.dashboard.health import collect_service_health, pending_service_health


settings = load_dashboard_settings()


@st.cache_data(ttl=settings.health_cache_seconds, show_spinner=False)
def _service_health():
    return collect_service_health(settings)


if st.sidebar.button("Refresh connections", use_container_width=True):
    _service_health.clear()
    st.session_state["dashboard_clear_policy_cache"] = True
    st.session_state.pop("dashboard_service_health", None)
    st.rerun()

statuses = st.session_state.get("dashboard_service_health")
if statuses is None:
    statuses = pending_service_health()
    st.session_state["dashboard_service_health"] = statuses
    st.sidebar.caption("Checking service availability…")
else:
    ready_count = sum(status.available for status in statuses)
    st.sidebar.caption(f"{ready_count} of {len(statuses)} services available")

# Render the selected page before unrelated health checks. Streamlit can therefore
# show useful content and the explicit pending state during a cold semantic-model load.
navigation.run()

if st.session_state.pop("dashboard_clear_policy_cache", False):
    from src.dashboard.policy_service import get_dashboard_policy_service

    get_dashboard_policy_service.cache_clear()

resolved_statuses = _service_health()
if resolved_statuses != statuses:
    st.session_state["dashboard_service_health"] = resolved_statuses
    st.rerun()

if initializing:
    startup_notice.empty()
