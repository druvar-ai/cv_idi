"""Streamlit UI Package."""

from .components import apply_industrial_theme, render_metric_card, get_status_badge_html
from .page_dashboard import render_dashboard
from .page_bare_pcb import render_bare_pcb_page
from .page_assembled_pcb import render_pcba_page
from .page_history import render_history_page
from .page_system_info import render_system_info_page

__all__ = [
    "apply_industrial_theme",
    "render_metric_card",
    "get_status_badge_html",
    "render_dashboard",
    "render_bare_pcb_page",
    "render_pcba_page",
    "render_history_page",
    "render_system_info_page",
]
