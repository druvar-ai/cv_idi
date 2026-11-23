"""Intelligent PCB Defect Inspection System — Main Application Entry Point."""

from pathlib import Path
import streamlit as st

from app.config import get_config
from app.storage.history_manager import HistoryManager
from app.ui import (
    apply_industrial_theme,
    render_dashboard,
    render_bare_pcb_page,
    render_pcba_page,
    render_history_page,
    render_system_info_page,
)


def main() -> None:
    """Main application loop."""
    st.set_page_config(
        page_title="Intelligent PCB Defect Inspection System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize master config & persistent database
    cfg = get_config()
    history = HistoryManager(db_path=Path("outputs/history/inspection_history.db"))

    # Apply industrial custom styling
    apply_industrial_theme()

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/fluency/96/integrated-circuit.png", width=64)
    st.sidebar.title("PCB Inspection AOI")
    st.sidebar.caption("Classical Computer Vision — Phase 1")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation Menu",
        [
            "Dashboard",
            "Bare PCB Inspection (Module A)",
            "Assembled PCB Inspection (Module B)",
            "Inspection History",
            "System Information & Specs"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        **Project Identity**  
        • **System:** Industrial AOI System  
        • **Course:** ACSC502 Computer Vision  
        • **Semester:** 5th Semester  
        • **Techniques:** Classical CV (OpenCV)  
        • **Status:** Operational  
    """)

    # Route to selected page
    if page == "Dashboard":
        render_dashboard(history)
    elif page == "Bare PCB Inspection (Module A)":
        render_bare_pcb_page(history)
    elif page == "Assembled PCB Inspection (Module B)":
        render_pcba_page(history)
    elif page == "Inspection History":
        render_history_page(history)
    elif page == "System Information & Specs":
        render_system_info_page()


if __name__ == "__main__":
    main()
