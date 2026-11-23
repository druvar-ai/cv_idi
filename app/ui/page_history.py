"""Page 4: Inspection History and Audit Log Explorer."""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from app.storage.history_manager import HistoryManager


def render_history_page(history: HistoryManager) -> None:
    """Render historical records with domain filtering and CSV export."""
    st.markdown('<div class="main-header">Inspection Audit Trail & History</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Review historical AOI runs, filter by defect category, and download batch telemetry</div>', unsafe_allow_html=True)

    # Filter Bar
    f1, f2, f3 = st.columns([1.5, 1.5, 2])
    with f1:
        domain_filter = st.selectbox("Inspection Domain:", ["All", "Bare PCB Fabrication", "Assembled PCB (PCBA)"])
    with f2:
        verdict_filter = st.selectbox("Inspection Result:", ["All", "PASS", "FAIL", "REVIEW"])
    with f3:
        search_query = st.text_input("Search by Inspection ID:", placeholder="e.g. INSP-BARE-00041000")

    # Fetch Records
    dom_arg = None if domain_filter == "All" else domain_filter
    ver_arg = None if verdict_filter == "All" else verdict_filter
    records = history.list_inspections(limit=100, inspection_type=dom_arg, decision=ver_arg)

    if search_query.strip():
        records = [r for r in records if search_query.strip().lower() in r["inspection_id"].lower()]

    if not records:
        st.info("No matching inspection records found.")
        return

    st.write(f"Displaying **{len(records)}** inspection record(s):")

    table_data = []
    for r in records:
        table_data.append({
            "Inspection ID": r["inspection_id"],
            "Timestamp (UTC)": r["timestamp"][:19].replace("T", " "),
            "Domain": r["inspection_type"],
            "Result": r["decision"],
            "Defect Count": r["total_defect_count"],
            "Latency": f"{r['processing_time_ms']:.1f} ms",
            "Reason": r["decision_reason"]
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # CSV Download of Filtered Table
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Export Filtered History as CSV", csv_bytes, file_name="aoi_inspection_history.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Granular Defect Inspection Details")
    sel_id = st.selectbox("Select Inspection ID to view defect breakdown:", [r["inspection_id"] for r in records])
    matched = next((r for r in records if r["inspection_id"] == sel_id), None)
    if matched:
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Domain:** {matched['inspection_type']}")
            st.write(f"**Result:** {matched['decision']}")
            st.write(f"**Reason:** {matched['decision_reason']}")
        with c2:
            st.write(f"**Total Defects / Anomalies:** {matched['total_defect_count']}")
            st.write(f"**Latency:** {matched['processing_time_ms']:.2f} ms")
            st.write(f"**Quality Inliers:** {matched['quality_metrics'].get('registration_inliers', 'N/A')}")

        if matched["defects"]:
            st.write("#### Detected Defect Items:")
            d_df = pd.DataFrame(matched["defects"])
            st.dataframe(d_df, use_container_width=True)
        else:
            st.success("Zero defect items recorded for this run.")
