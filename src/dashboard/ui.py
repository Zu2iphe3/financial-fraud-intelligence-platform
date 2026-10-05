"""Shared presentation helpers for a consistent restrained dashboard."""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from src.dashboard.health import ServiceHealth


def apply_dashboard_styles() -> None:
    """Add only layout polish that Streamlit's theme cannot express."""

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1180px;}
        [data-testid="stSidebar"] {min-width: 270px; max-width: 270px;}
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: 0.55rem;}
        [data-testid="stMetric"] {
            border: 1px solid #D7E1E3;
            border-radius: 0.65rem;
            padding: 0.72rem 0.85rem;
            background: #FFFFFF;
            box-shadow: 0 1px 2px rgba(23, 36, 43, 0.04);
        }
        [data-testid="stMetricLabel"] {font-weight: 600;}
        [data-testid="stMetricValue"] {font-size: 1.62rem;}
        [data-testid="stVerticalBlock"] {gap: 0.65rem;}
        h1 {font-size: 2.15rem !important; letter-spacing: -0.025em; margin-bottom: 0.15rem;}
        h2, h3 {letter-spacing: -0.015em;}
        .ffip-kicker {color: #1F6E68; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;}
        .ffip-muted {color: #52636C; margin-bottom: 0.45rem; max-width: 880px;}
        .ffip-health-strip {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            flex-wrap: wrap;
            padding: 0.55rem 0;
            margin: 0.1rem 0 0.35rem;
            border-top: 1px solid #DFE7E9;
            border-bottom: 1px solid #DFE7E9;
        }
        .ffip-health-label {
            color: #52636C;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-right: 0.15rem;
            text-transform: uppercase;
        }
        .ffip-health-item {
            display: inline-flex;
            align-items: center;
            gap: 0.32rem;
            color: #33454D;
            font-size: 0.82rem;
            white-space: nowrap;
        }
        .ffip-health-dot {width: 0.48rem; height: 0.48rem; border-radius: 50%; display: inline-block;}
        .ffip-health-dot.available {background: #18864B;}
        .ffip-health-dot.unavailable {background: #C53B3B;}
        .ffip-health-dot.setup_required {background: #C07A12;}
        .ffip-health-dot.checking {background: #7A8A91;}
        .ffip-scope {color: #52636C; font-size: 0.87rem; padding-top: 0.45rem;}
        [data-testid="stTabs"] button {font-weight: 600;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, description: str) -> None:
    st.markdown(
        '<p class="ffip-kicker">Financial Fraud Intelligence Platform</p>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(f'<p class="ffip-muted">{description}</p>', unsafe_allow_html=True)


def render_health_strip(statuses: Iterable["ServiceHealth"]) -> None:
    """Present service readiness as one compact line above the analytics."""

    items = "".join(
        (
            '<span class="ffip-health-item" '
            f'title="{escape(status.message)}">'
            f'<span class="ffip-health-dot {status.state}"></span>'
            f"{escape(status.label)}</span>"
        )
        for status in statuses
    )
    st.markdown(
        f'<div class="ffip-health-strip"><span class="ffip-health-label">Services</span>{items}</div>',
        unsafe_allow_html=True,
    )
