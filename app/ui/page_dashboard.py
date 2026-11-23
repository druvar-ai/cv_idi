"""Page 1: Industrial AOI Executive Dashboard."""

import pandas as pd
import streamlit as st

from app.storage.history_manager import HistoryManager
from app.ui.components import render_metric_card, get_status_badge_html


def render_dashboard(history: HistoryManager) -> None:
    """Render high-level production metrics and recent inspection logs."""
    st.markdown('<div class="main-header">AOI Executive Production Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Optical Inspection System — Phase 1 Classical Machine Vision Telemetry</div>', unsafe_allow_html=True)

    stats = history.get_summary_stats()

    # KPI Top Bar
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card("Total Inspected", str(stats["total_inspections"]), "Cumulative runs")
    with c2:
        render_metric_card("Bare PCB Runs", str(stats["bare_pcb_inspections"]), "Fabrication stage")
    with c3:
        render_metric_card("PCBA Runs", str(stats["assembled_pcb_inspections"]), "Assembly stage")
    with c4:
        rate = f"{stats['pass_rate_percent']:.1f}%" if stats["total_inspections"] > 0 else "N/A"
        render_metric_card("First-Pass Yield", rate, f"{stats['pass_count']} Passed")
    with c5:
        avg_lat = f"{stats['avg_latency_ms']:.1f} ms" if stats["total_inspections"] > 0 else "0.0 ms"
        render_metric_card("Avg Latency", avg_lat, "Per board cycle")

    st.markdown("<br>", unsafe_allow_html=True)

    # Secondary Breakdown Cards
    b1, b2, b3 = st.columns(3)
    with b1:
        st.info(f"**Passed Units:** {stats['pass_count']} units released without defects.")
    with b2:
        st.error(f"**Defective / Rejected Units:** {stats['fail_count']} units flagged for rework.")
    with b3:
        st.warning(f"**Engineering Review:** {stats['review_count']} units held for manual inspection.")

    st.markdown("---")

    # Recent Inspections Table
    st.subheader("Recent Automated Inspections")
    records = history.list_inspections(limit=15)

    if not records:
        st.info("No inspections recorded yet. Navigate to 'Bare PCB Inspection' or 'Assembled PCB Inspection' to run tests.")
    else:
        table_rows = []
        for r in records:
            table_rows.append({
                "Inspection ID": r["inspection_id"],
                "Timestamp (UTC)": r["timestamp"][:19].replace("T", " "),
                "Domain": r["inspection_type"],
                "Result": r["decision"],
                "Defect Count": r["total_defect_count"],
                "Processing Time": f"{r['processing_time_ms']:.1f} ms",
                "Reason": r["decision_reason"]
            })
        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
