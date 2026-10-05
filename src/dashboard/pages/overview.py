"""Overview and historical fraud-monitoring analytics page."""

from __future__ import annotations

import streamlit as st

from src.dashboard.ui import render_health_strip, render_page_header


TABLE_ROW_HEIGHT = 35
TABLE_HEADER_HEIGHT = 35
DAILY_TABLE_VISIBLE_ROWS = 8
TYPE_TABLE_VISIBLE_ROWS = 5
DAILY_TABLE_HEIGHT = TABLE_HEADER_HEIGHT + TABLE_ROW_HEIGHT * DAILY_TABLE_VISIBLE_ROWS
TYPE_TABLE_HEIGHT = TABLE_HEADER_HEIGHT + TABLE_ROW_HEIGHT * TYPE_TABLE_VISIBLE_ROWS
AGGREGATE_TABLE_COLUMN_WIDTHS = (1, 1)


@st.cache_resource
def _repository():
    # Keep the first page frame independent of pandas/psycopg import time.
    from src.dashboard.analytics import AnalyticsRepository
    from src.data.postgres import load_postgres_settings

    return AnalyticsRepository(load_postgres_settings())


@st.cache_data(ttl=30, show_spinner=False)
def _filter_options() -> tuple[int, int, tuple[str, ...]]:
    return _repository().load_filter_options()


@st.cache_data(ttl=30, show_spinner=False)
def _snapshot(start_day: int, end_day: int, transaction_types: tuple[str, ...]):
    from src.dashboard.analytics import AnalyticsFilters

    return _repository().load_snapshot(
        AnalyticsFilters(start_day, end_day, transaction_types)
    )


def _render_metrics(snapshot) -> None:
    metrics = snapshot.metrics
    columns = st.columns(4)
    columns[0].metric("Transaction volume", f"{metrics.transaction_count:,}")
    columns[1].metric("Labelled fraud cases", f"{metrics.fraud_count:,}")
    columns[2].metric("Labelled fraud rate", f"{metrics.fraud_rate:.3%}")
    columns[3].metric("Labelled fraud value", _compact_number(metrics.fraud_amount))
    columns[3].caption("simulated monetary units")


def _compact_number(value: float) -> str:
    """Shorten large display values without changing the underlying calculation."""

    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:,.2f}"


def _aggregated_display_tables(daily, by_type):
    """Expose readable aggregates while hiding duplicate raw-rate representations."""

    daily_display = daily[
        [
            "simulation_day",
            "transaction_count",
            "fraud_count",
            "fraud_rate_percent",
            "is_partial_day",
        ]
    ].rename(
        columns={
            "simulation_day": "Day",
            "transaction_count": "Transactions",
            "fraud_count": "Labelled fraud cases",
            "fraud_rate_percent": "Labelled fraud rate (%)",
            "is_partial_day": "Partial day",
        }
    )
    type_display = by_type[
        [
            "transaction_type",
            "transaction_count",
            "fraud_count",
            "fraud_rate_percent",
        ]
    ].rename(
        columns={
            "transaction_type": "Transaction type",
            "transaction_count": "Transactions",
            "fraud_count": "Labelled fraud cases",
            "fraud_rate_percent": "Labelled fraud rate (%)",
        }
    )
    daily_display["Labelled fraud rate (%)"] = daily_display[
        "Labelled fraud rate (%)"
    ].round(4)
    type_display["Labelled fraud rate (%)"] = type_display[
        "Labelled fraud rate (%)"
    ].round(4)
    return daily_display, type_display


def _render_charts(snapshot) -> None:
    daily = snapshot.daily.copy()
    by_type = snapshot.by_transaction_type.copy()
    if daily.empty or by_type.empty:
        st.info("No trusted mart records match the selected filters.")
        return

    daily["fraud_rate_percent"] = daily["fraud_rate"].astype(float) * 100
    by_type["fraud_rate_percent"] = by_type["fraud_rate"].astype(float) * 100

    if daily["is_partial_day"].astype(bool).any():
        st.warning(
            "Day 31 is a partial simulation day: it contains only 23 observed hours and "
            "272 transactions. Its artificial 100% labelled fraud rate is not a "
            "representative fraud pattern."
        )

    daily_tab, type_tab, data_tab = st.tabs(
        ["Daily Patterns", "Transaction Types", "Aggregated Data"]
    )
    with daily_tab:
        volume_column, fraud_column, rate_column = st.columns(3)
        with volume_column:
            st.caption("Observed transaction volume by simulation day")
            st.line_chart(
                daily,
                x="simulation_day",
                y="transaction_count",
                x_label="Simulation day",
                y_label="Observed transactions",
                height=280,
            )
        with fraud_column:
            st.caption("Labelled fraud cases by simulation day")
            st.line_chart(
                daily,
                x="simulation_day",
                y="fraud_count",
                x_label="Simulation day",
                y_label="Labelled fraud cases",
                height=280,
            )
        with rate_column:
            st.caption("Labelled fraud rate by simulation day")
            st.line_chart(
                daily,
                x="simulation_day",
                y="fraud_rate_percent",
                x_label="Simulation day",
                y_label="Labelled fraud rate (%)",
                height=280,
            )
    with type_tab:
        count_column, type_rate_column = st.columns(2)
        with count_column:
            st.caption("Labelled fraud cases by transaction type")
            st.bar_chart(
                by_type,
                x="transaction_type",
                y="fraud_count",
                x_label="Transaction type",
                y_label="Labelled fraud cases",
                height=300,
            )
        with type_rate_column:
            st.caption("Labelled fraud rate by transaction type")
            st.bar_chart(
                by_type,
                x="transaction_type",
                y="fraud_rate_percent",
                x_label="Transaction type",
                y_label="Labelled fraud rate (%)",
                height=300,
            )
    with data_tab:
        st.caption("Read-only PaySim aggregates for the selected historical view.")
        daily_display, type_display = _aggregated_display_tables(daily, by_type)
        # Equal space keeps every transaction-type heading and value readable while
        # preserving the compact side-by-side aggregate layout.
        daily_column, type_column = st.columns(AGGREGATE_TABLE_COLUMN_WIDTHS)
        with daily_column:
            st.markdown("**Daily aggregates**")
            st.dataframe(
                daily_display,
                hide_index=True,
                width="stretch",
                height=DAILY_TABLE_HEIGHT,
                row_height=TABLE_ROW_HEIGHT,
            )
        with type_column:
            st.markdown("**Transaction-type aggregates**")
            st.dataframe(
                type_display,
                hide_index=True,
                width="stretch",
                height=TYPE_TABLE_HEIGHT,
                row_height=TABLE_ROW_HEIGHT,
            )


def render_page() -> None:
    render_page_header(
        "Overview & Analytics",
        "Explore observed transaction activity and labelled fraud patterns from the "
        "simulated PaySim dataset. These descriptive analytics show source-data outcomes "
        "and are separate from ML fraud-risk predictions.",
    )
    statuses = st.session_state.get("dashboard_service_health", ())
    render_health_strip(statuses)

    # A cold start first renders the complete page frame, then app.py performs the
    # real dependency checks and reruns. This prevents a slow database or embedding
    # model from leaving the user with an incomplete page that needs interaction.
    if statuses and all(status.state == "checking" for status in statuses):
        st.info(
            "Checking dashboard services. Historical fraud intelligence will load "
            "automatically when the connection checks finish."
        )
        return

    from src.dashboard.analytics import AnalyticsUnavailableError

    try:
        min_day, max_day, transaction_types = _filter_options()
    except (AnalyticsUnavailableError, ValueError):
        st.error(
            "Historical analytics are unavailable. Prepare the analytics data, or continue "
            "using risk scoring and policy guidance independently."
        )
        return

    if st.session_state.pop("analytics_reset_requested", False):
        st.session_state["analytics_day_range"] = (min_day, max_day)
        st.session_state["analytics_transaction_types"] = list(transaction_types)

    scope_column, filter_column = st.columns([5, 1])
    with filter_column:
        with st.popover("Filters", use_container_width=True):
            with st.form("analytics_filters"):
                st.caption("Refine the historical view")
                day_range = st.slider(
                    "Simulation days",
                    min_value=min_day,
                    max_value=max_day,
                    value=(min_day, max_day),
                    key="analytics_day_range",
                )
                selected_types = st.multiselect(
                    "Transaction types",
                    transaction_types,
                    default=list(transaction_types),
                    key="analytics_transaction_types",
                )
                apply_filters = st.form_submit_button(
                    "Apply", type="primary", use_container_width=True
                )
                reset_filters = st.form_submit_button("Reset", use_container_width=True)
    with scope_column:
        st.markdown(
            f'<p class="ffip-scope">Simulation days {day_range[0]}–{day_range[1]} · '
            f"{len(selected_types)} transaction types selected</p>",
            unsafe_allow_html=True,
        )

    if reset_filters:
        st.session_state["analytics_reset_requested"] = True
        st.rerun()

    if not selected_types:
        st.info("Select at least one transaction type to display analytics.")
        return
    with st.spinner("Loading fraud intelligence..."):
        try:
            snapshot = _snapshot(day_range[0], day_range[1], tuple(selected_types))
        except AnalyticsUnavailableError:
            st.error("Historical analytics could not be loaded. Other pages remain available.")
            return
    _render_metrics(snapshot)
    _render_charts(snapshot)


if __name__ == "__main__":
    render_page()
