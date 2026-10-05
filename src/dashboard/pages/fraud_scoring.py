"""Operational transaction investigation backed by point-in-time model scoring."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.dashboard.config import load_dashboard_settings
from src.dashboard.model_information import (
    ModelEvaluationInformation,
    ModelInformationError,
    load_model_information,
)
from src.dashboard.prediction_client import (
    PredictionClient,
    PredictionRejectedError,
    PredictionServiceUnavailable,
)
from src.dashboard.transaction_scoring import (
    HistoricalTransactionScore,
    HistoricalTransactionScoringService,
)
from src.dashboard.ui import render_page_header
from src.data.postgres import load_postgres_settings
from src.features.point_in_time import (
    FeatureHistoryUnavailableError,
    PointInTimeFeatureError,
    PointInTimeFeatureService,
    PreparedTransactionFeatures,
    ReviewTransaction,
    TransactionNotFoundError,
)


@st.cache_resource
def _transaction_scoring_service() -> HistoricalTransactionScoringService:
    """Reuse lightweight clients while each feature lookup opens its own DB session."""
    dashboard = load_dashboard_settings()
    return HistoricalTransactionScoringService(
        PointInTimeFeatureService(load_postgres_settings()),
        PredictionClient(
            dashboard.prediction_api_base_url,
            dashboard.request_timeout_seconds,
        ),
    )


@st.cache_data(ttl=60, show_spinner=False)
def _review_transactions() -> tuple[ReviewTransaction, ...]:
    """Keep the operational queue responsive without caching prediction results."""
    return _transaction_scoring_service().list_review_transactions(20)


@st.cache_data(show_spinner=False)
def _model_information():
    """Load the immutable reviewed evidence once per dashboard process."""
    return load_model_information()


def _count_from_log1p(value: float) -> int:
    """Present exact historical counts without exposing the model transform."""
    return max(0, int(round(math.expm1(float(value)))))


def _amount_from_log1p(value: float) -> float:
    """Undo display-only log scaling while tolerating float32 rounding near zero."""
    return max(0.0, math.expm1(float(value)))


def _history_summary(prepared: PreparedTransactionFeatures) -> dict[str, float | int | str]:
    """Translate model-ready history into investigator-friendly values."""
    row = prepared.features.iloc[0]
    history_available = int(row["dest_history_available"]) == 1
    return {
        "transactions_24h": _count_from_log1p(row["dest_tx_count_log1p_24h"]),
        "transactions_7d": _count_from_log1p(row["dest_tx_count_log1p_168h"]),
        "amount_24h": _amount_from_log1p(row["dest_amount_sum_log1p_24h"]),
        "amount_7d": _amount_from_log1p(row["dest_amount_sum_log1p_168h"]),
        "unique_origins_7d": _count_from_log1p(
            row["dest_unique_origins_log1p_168h"]
        ),
        "previous_activity": (
            f"{int(round(float(row['dest_hours_since_previous_activity'])))} hours ago"
            if history_available
            else "No earlier activity"
        ),
    }


def _technical_feature_table(prepared: PreparedTransactionFeatures) -> pd.DataFrame:
    """Create a read-only table that preserves contract order and dtypes."""
    row = prepared.features.iloc[0]
    return pd.DataFrame(
        {
            "Feature": list(prepared.features.columns),
            "Value": [float(row[name]) for name in prepared.features.columns],
            "Data type": [str(prepared.features[name].dtype) for name in prepared.features],
        }
    )


def _render_transaction(prepared: PreparedTransactionFeatures) -> None:
    with st.container(border=True):
        st.subheader("Transaction under review")
        st.caption("Transaction ID")
        st.code(prepared.transaction_id, language=None)

        columns = st.columns(4)
        columns[0].metric("Type", prepared.transaction_type)
        columns[1].metric("Amount", f"{prepared.amount:,.2f}")
        columns[2].metric("Simulation day", f"Day {prepared.simulation_day}")
        columns[3].metric("Simulation step", f"{prepared.step}")

        origin, destination = st.columns(2)
        origin.markdown(f"**Origin**  \n`{prepared.name_orig}`")
        destination.markdown(f"**Destination**  \n`{prepared.name_dest}`")
        st.caption(
            f"Simulated hour {prepared.simulation_hour_of_day} of day "
            f"{prepared.simulation_day} · source row {prepared.source_row_number:,}"
        )


def _render_history(prepared: PreparedTransactionFeatures) -> None:
    summary = _history_summary(prepared)
    with st.container(border=True):
        st.subheader("Destination activity before this transaction")
        columns = st.columns(4)
        columns[0].metric("Transactions · 24h", f"{summary['transactions_24h']:,}")
        columns[1].metric("Transactions · 7 days", f"{summary['transactions_7d']:,}")
        columns[2].metric("Unique origins · 7 days", f"{summary['unique_origins_7d']:,}")
        columns[3].metric("Previous activity", str(summary["previous_activity"]))
        st.caption(
            f"Earlier destination value: {summary['amount_24h']:,.2f} over 24 hours · "
            f"{summary['amount_7d']:,.2f} over 7 days. Same-step and future "
            "transactions are excluded."
        )


def _render_risk(result: HistoricalTransactionScore) -> None:
    prediction = result.prediction
    risk = prediction.risk_classification
    guidance = {
        "high": "Prioritize this transaction for analyst review.",
        "moderate": "Review the transaction alongside other available evidence.",
        "low": "Lower model-estimated risk; retain normal monitoring controls.",
    }[risk]

    with st.container(border=True):
        st.subheader("Risk assessment")
        probability, classification = st.columns([1, 2])
        probability.metric("Fraud probability", f"{prediction.fraud_probability:.2%}")
        message = f"{risk.title()} risk · {guidance}"
        if risk == "high":
            classification.error(message)
        elif risk == "moderate":
            classification.warning(message)
        else:
            classification.success(message)
        st.write(
            "This score supports human investigation. It does not by itself determine "
            "that fraud occurred."
        )
        st.caption(f"Model version: {prediction.model_version}")


def _render_technical_details(result: HistoricalTransactionScore) -> None:
    prediction = result.prediction
    with st.expander("Technical details"):
        st.caption(
            "Read-only model inputs reconstructed from activity strictly before the "
            "selected transaction."
        )
        st.dataframe(
            _technical_feature_table(result.prepared),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Feature": st.column_config.TextColumn("Model feature", width="large"),
                "Value": st.column_config.NumberColumn("Value", format="%.6f"),
                "Data type": st.column_config.TextColumn("Type", width="small"),
            },
        )
        st.caption(
            f"Risk policy: {prediction.risk_policy_version} · Feature contract: "
            "paysim_features_v2"
        )


def _render_model_information(result: HistoricalTransactionScore) -> None:
    """Show only reviewed validation evidence for the model that returned the score."""
    try:
        information = _model_information()
    except ModelInformationError:
        return
    if information.model_version != result.prediction.model_version:
        return

    with st.expander("Model information"):
        identity, threshold = st.columns([3, 1])
        identity.markdown(f"**{information.model_type}**")
        identity.caption(information.model_version)
        threshold.metric("High-risk threshold", f"{information.decision_threshold:.2%}")
        threshold.caption(information.threshold_status)

        _render_evaluation_summary(information.validation_evaluation)
        st.divider()
        _render_evaluation_summary(information.held_out_test_evaluation)


def _render_evaluation_summary(evaluation: ModelEvaluationInformation) -> None:
    """Keep ranking and frozen-threshold evidence distinct and compact."""
    st.markdown(f"**{evaluation.label}**")
    st.caption(f"{evaluation.population} · {evaluation.role}")
    metrics = st.columns(5)
    metrics[0].metric("PR-AUC · ranking", f"{evaluation.pr_auc:.3f}")
    metrics[1].metric("ROC-AUC · ranking", f"{evaluation.roc_auc:.3f}")
    metrics[2].metric("Precision · frozen", f"{evaluation.precision:.1%}")
    metrics[3].metric("Recall · frozen", f"{evaluation.recall:.1%}")
    metrics[4].metric("F1 · frozen", f"{evaluation.f1:.3f}")
    st.caption(
        f"Population: {evaluation.rows:,} transactions · {evaluation.fraud_rows:,} "
        f"fraud ({evaluation.fraud_rate:.2%}) · {evaluation.legitimate_rows:,} "
        f"legitimate · accuracy {evaluation.accuracy:.2%}. Confusion matrix at "
        f"{evaluation.threshold:.6f}: TP {evaluation.true_positives:,} · "
        f"FP {evaluation.false_positives:,} · FN {evaluation.false_negatives:,} · "
        f"TN {evaluation.true_negatives:,}."
    )


def _review_label(transaction: ReviewTransaction) -> str:
    """Format a compact selection label while retaining the full UUID internally."""
    return (
        f"Day {transaction.simulation_day}, hour {transaction.simulation_hour_of_day} · "
        f"{transaction.transaction_type} · {transaction.amount:,.2f} · "
        f"{transaction.short_transaction_id}"
    )


def _score_transaction(transaction_id: str) -> None:
    """Run both discovery routes through the one reviewed scoring pipeline."""
    st.session_state.pop("historical_transaction_score", None)
    if not transaction_id:
        st.warning("Enter or select a transaction to begin the analysis.")
        return
    try:
        with st.spinner(
            "Loading the transaction, reconstructing prior activity, and scoring risk..."
        ):
            result = _transaction_scoring_service().score(transaction_id)
        st.session_state["historical_transaction_score"] = result
    except TransactionNotFoundError:
        st.error("The transaction ID is invalid or was not found in PostgreSQL.")
    except FeatureHistoryUnavailableError:
        st.error(
            "Transaction history is unavailable. Confirm PostgreSQL and the trusted "
            "transaction mart are ready."
        )
    except PointInTimeFeatureError:
        st.error(
            "The transaction history could not be prepared safely, so no score was "
            "generated."
        )
    except PredictionRejectedError:
        st.error("The prediction service rejected the reconstructed feature vector.")
    except PredictionServiceUnavailable:
        st.error(
            "The prediction service is unavailable. No fraud-risk score was generated."
        )


def render_page() -> None:
    render_page_header(
        "Fraud Risk Scoring",
        "Investigate a stored transaction using its activity history and the reviewed "
        "fraud-risk model.",
    )

    with st.container(border=True):
        st.subheader("Transactions for review")
        st.caption("Browse transactions stored in the trusted transaction mart.")
        if st.button("Load transactions", key="load_review_transactions"):
            try:
                with st.spinner("Loading transactions for review..."):
                    st.session_state["review_transactions"] = _review_transactions()
            except FeatureHistoryUnavailableError:
                st.error(
                    "The transaction list is unavailable. You can still look up a known "
                    "transaction ID below."
                )

        review_transactions = st.session_state.get("review_transactions", ())
        if review_transactions:
            selected = st.selectbox(
                "Stored transaction",
                review_transactions,
                format_func=_review_label,
                key="review_transaction_selection",
                label_visibility="collapsed",
            )
            if st.button(
                "Analyse selected transaction",
                type="primary",
                use_container_width=True,
            ):
                _score_transaction(selected.transaction_id)

    st.caption("Or look up a specific transaction")
    with st.form("fraud_scoring_form"):
        transaction_id = st.text_input(
            "Transaction ID",
            placeholder="Enter a PostgreSQL transaction UUID",
            help="Use the transaction_id stored in the trusted transaction mart.",
        ).strip()
        submitted = st.form_submit_button(
            "Analyse risk", type="primary", use_container_width=True
        )

    if submitted:
        _score_transaction(transaction_id)

    result = st.session_state.get("historical_transaction_score")
    if result:
        _render_transaction(result.prepared)
        _render_risk(result)
        _render_history(result.prepared)
        _render_technical_details(result)
        _render_model_information(result)


if __name__ == "__main__":
    render_page()
