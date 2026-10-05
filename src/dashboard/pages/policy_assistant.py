"""Grounded fictional policy-assistant page."""

from __future__ import annotations

import streamlit as st

from src.dashboard.ui import render_page_header


EXAMPLE_QUESTIONS = (
    "What should an analyst investigate when a transfer goes to a new destination?",
    "When should rapid activity involving one destination be escalated?",
    "Why is a fraud probability not proof of fraud?",
)


def _policy_service():
    # Health and the query page share this initialization, so green search status
    # means this exact embedding/vector-store path already completed successfully.
    from src.dashboard.policy_service import get_dashboard_policy_service

    return get_dashboard_policy_service()


def _render_sources(sources) -> None:
    st.subheader("Fictional policy sources")
    for source in sources:
        with st.expander(f"{source.title} — {source.section_id}"):
            st.write(source.section_heading)
            st.caption(f"Source: {source.source_path} · fictional portfolio corpus")


def _apply_example_question() -> None:
    """Copy an example only when the selector changes, preserving later custom edits."""

    selected = st.session_state.get("policy_example")
    if selected:
        st.session_state["policy_question"] = selected


def render_page() -> None:
    render_page_header(
        "Policy Assistant",
        "Ask about investigation, escalation, and interpreting fraud risk using the "
        "fictional policy library. This guidance is separate from scoring and is not legal advice.",
    )
    st.caption(
        "The assistant answers only from the policy library. Verified sources are shown "
        "separately from the generated explanation."
    )

    st.selectbox(
        "Example question",
        EXAMPLE_QUESTIONS,
        index=None,
        placeholder="Choose an example or write your own",
        key="policy_example",
        on_change=_apply_example_question,
    )

    with st.form("policy_question_form"):
        question = st.text_area(
            "Question",
            key="policy_question",
            placeholder="Ask about investigation, escalation, or interpreting fraud risk...",
            height=110,
        )
        submitted = st.form_submit_button(
            "Ask policy assistant", type="primary", use_container_width=True
        )

    if submitted:
        # Loading Transformers/Chroma is intentionally deferred until after the full
        # page shell and question form are visible on a fresh process.
        from src.dashboard.policy_service import (
            PolicyGenerationUnavailable,
            PolicyResponseRejected,
            PolicySearchUnavailable,
        )

        if not question.strip():
            st.warning("Enter a supported policy question before submitting.")
        else:
            try:
                with st.status("Searching the policy library...", expanded=True) as progress:
                    st.write("Loading local semantic search...")
                    service = _policy_service()
                    st.write("Reviewing relevant evidence and preparing a grounded answer...")
                    answer = service.ask(question)
                    progress.update(label="Policy review complete", state="complete")
                st.session_state["policy_answer"] = answer
            except PolicySearchUnavailable:
                st.session_state.pop("policy_answer", None)
                st.error(
                    "Policy search could not start. The policy library may be present, but "
                    "the local search model is not ready. Check the local embedding setup "
                    "and try again; analytics and scoring remain usable."
                )
            except PolicyGenerationUnavailable:
                st.session_state.pop("policy_answer", None)
                st.error(
                    "Relevant policy guidance was found, but the local answer generator "
                    "did not respond. Check its service status and try again."
                )
            except PolicyResponseRejected:
                st.session_state.pop("policy_answer", None)
                st.warning(
                    "The generated response was blocked because it did not meet the "
                    "assistant's grounding safeguards."
                )
            except ValueError:
                st.session_state.pop("policy_answer", None)
                st.warning("Enter a valid policy question and try again.")

    answer = st.session_state.get("policy_answer")
    if answer:
        with st.container(border=True):
            st.subheader("Grounded guidance")
            if answer.status == "insufficient_evidence":
                st.warning(answer.answer)
            else:
                st.write(answer.answer)
        if answer.sources:
            _render_sources(answer.sources)


if __name__ == "__main__":
    render_page()
