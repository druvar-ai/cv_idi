"""Reusable UI Components and Styling Helpers for Industrial AOI Application."""

from typing import Dict, List, Optional
import streamlit as st


def apply_industrial_theme() -> None:
    """Inject clean industrial machine vision styling."""
    st.markdown("""
        <style>
        /* Modern Industrial Machine Vision Styling */
        .main-header {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 0.95rem;
            color: #64748b;
            margin-bottom: 1.2rem;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            text-align: center;
        }
        .metric-label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            color: #64748b;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0f172a;
            margin: 4px 0;
        }
        .badge-pass {
            background-color: #dcfce7;
            color: #15803d;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-fail {
            background-color: #fee2e2;
            color: #b91c1c;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-review {
            background-color: #fef3c7;
            color: #b45309;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 0.85rem;
            display: inline-block;
        }
        </style>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = "") -> None:
    """Render a polished KPI metric card."""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div style="font-size: 0.75rem; color: #94a3b8;">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)


def get_status_badge_html(status: str) -> str:
    """Return colored pill HTML for PASS, FAIL, or REVIEW."""
    s = status.upper()
    if s == "PASS":
        return '<span class="badge-pass">PASS</span>'
    elif s == "FAIL":
        return '<span class="badge-fail">FAIL</span>'
    else:
        return '<span class="badge-review">REVIEW</span>'
