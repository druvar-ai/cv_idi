"""Page 3: Interactive Assembled PCB (PCBA) Inspection."""

import glob
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app.acquisition.loader import ImageLoader
from app.config import get_config
from app.inspection.assembled_pcb.pipeline import PCBAInspectionPipeline
from app.inspection.decision_engine import UnifiedDecisionEngine
from app.reporting.json_reporter import JSONReporter
from app.reporting.csv_reporter import CSVReporter
from app.reporting.pdf_reporter import PDFReporter
from app.storage.history_manager import HistoryManager
from app.ui.components import get_status_badge_html


def render_pcba_page(history: HistoryManager) -> None:
    """Render interactive SMT component placement inspection page."""
    st.markdown('<div class="main-header">Assembled PCB (PCBA) Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Module B: Inspection of Missing, Shifted, and Rotated Surface-Mount Components</div>', unsafe_allow_html=True)

    cfg = get_config()
    pipeline = PCBAInspectionPipeline(config=cfg.pcba)
    engine = UnifiedDecisionEngine(config=cfg.decision_engine)

    # Input Mode Selection
    mode = st.radio("Acquisition Source:", ["Curated PCBA Industrial Benchmark", "Upload Custom PCBA Boards"], horizontal=True)

    ref_bgr = None
    test_bgr = None
    spec_path = None
    sample_id = "custom_pcba"

    if mode == "Curated PCBA Industrial Benchmark":
        base_dir = Path("datasets/assembled_pcb")
        ref_path = base_dir / "templates" / "template_pcba_01.jpg"
        spec_path = base_dir / "annotations" / "template_pcba_01.xml"

        test_opts = {
            "Defective Board (Missing 53_ac, Rotated 47/48, Shifted)": base_dir / "images" / "test_pcba_01_defective.jpg",
            "Defect-Free Reference Board (Pass Test)": ref_path
        }
        # Add additional error images if available
        err_files = sorted(glob.glob(str(base_dir / "images" / "test_pcba_err_*.jpg")))
        for ef in err_files:
            test_opts[f"Error Board: {Path(ef).name}"] = Path(ef)

        sel_test_label = st.selectbox("Select Test Assembly Board:", list(test_opts.keys()))
        selected_test_path = test_opts[sel_test_label]

        ref_img_obj, val_ref = ImageLoader.load_from_path(ref_path)
        test_img_obj, val_test = ImageLoader.load_from_path(selected_test_path)

        if ref_img_obj and test_img_obj:
            ref_bgr = ref_img_obj.image_bgr
            test_bgr = test_img_obj.image_bgr
            sample_id = selected_test_path.stem
    else:
        u1, u2 = st.columns(2)
        with u1:
            u_ref = st.file_uploader("Upload Reference PCBA Board", type=["jpg", "jpeg", "png"])
        with u2:
            u_test = st.file_uploader("Upload Assembled Test PCBA Board", type=["jpg", "jpeg", "png"])
        u_xml = st.file_uploader("Optional: Upload Component Specification (Pascal VOC XML)", type=["xml"])

        if u_ref and u_test:
            ref_img_obj, val_ref = ImageLoader.load_from_bytes(u_ref.read(), source_name=u_ref.name)
            test_img_obj, val_test = ImageLoader.load_from_bytes(u_test.read(), source_name=u_test.name)
            if ref_img_obj and test_img_obj:
                ref_bgr = ref_img_obj.image_bgr
                test_bgr = test_img_obj.image_bgr
                sample_id = Path(u_test.name).stem
                if u_xml:
                    temp_xml_p = Path("outputs/intermediate") / u_xml.name
                    temp_xml_p.write_bytes(u_xml.read())
                    spec_path = temp_xml_p

    if ref_bgr is not None and test_bgr is not None:
        st.success(f"PCBA Boards Loaded: Reference ({ref_bgr.shape[1]}x{ref_bgr.shape[0]}), Test ({test_bgr.shape[1]}x{test_bgr.shape[0]})")

        if st.button("Execute SMT Component Placement Inspection", type="primary", use_container_width=True):
            with st.spinner("Analyzing Component Alignment and Placement Anomalies..."):
                raw_res = pipeline.inspect(
                    reference_bgr=ref_bgr,
                    test_bgr=test_bgr,
                    components_spec=spec_path,
                    inspection_id=f"INSP-PCBA-{sample_id}"
                )
                verdict = engine.arbitrate_pcba(raw_res)
                history.add_inspection(verdict)

            # Display Verdict Summary
            st.markdown("---")
            v1, v2, v3, v4 = st.columns([1.5, 2.5, 1.5, 1.5])
            with v1:
                st.markdown(f"### Verdict: {get_status_badge_html(verdict.decision)}", unsafe_allow_html=True)
            with v2:
                st.markdown(f"**Decision Reason:** {verdict.decision_reason}")
            with v3:
                st.metric("Total Components", raw_res.total_components_inspected)
            with v4:
                st.metric("Placement Anomalies", raw_res.total_anomalies)

            # Visual Inspection Tabs
            tabs = st.tabs([
                "1. Placement Anomaly Bounding Boxes",
                "2. Photometric Difference Heatmap",
                "3. Nominal Component Grid Overlay",
                "4. Board-Level Registration",
                "5. Original Input Boards"
            ])

            with tabs[0]:
                st.image(cv2.cvtColor(verdict.visualizations["bbox_vis"], cv2.COLOR_BGR2RGB), caption="Inspection Map: Green=OK, Red=Missing, Orange=Shifted, Magenta=Rotated", use_container_width=True)

            with tabs[1]:
                st.image(cv2.cvtColor(verdict.visualizations["difference_image"], cv2.COLOR_BGR2RGB), caption="Photometric Difference Heatmap", use_container_width=True)

            with tabs[2]:
                st.image(cv2.cvtColor(verdict.visualizations["component_regions"], cv2.COLOR_BGR2RGB), caption="Nominal Component Inspection Grid", use_container_width=True)

            with tabs[3]:
                c_al1, c_al2 = st.columns(2)
                with c_al1:
                    st.image(cv2.cvtColor(verdict.visualizations["original_reference"], cv2.COLOR_BGR2RGB), caption="Nominal Reference Assembly", use_container_width=True)
                with c_al2:
                    st.image(cv2.cvtColor(verdict.visualizations["aligned_test"], cv2.COLOR_BGR2RGB), caption=f"Homography-Aligned Test Board (Inliers: {verdict.quality_metrics.get('registration_inliers', 1.0)*100:.1f}%)", use_container_width=True)

            with tabs[4]:
                c_in1, c_in2 = st.columns(2)
                with c_in1:
                    st.image(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB), caption="Reference Assembly Board", use_container_width=True)
                with c_in2:
                    st.image(cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB), caption="Test Assembly Board", use_container_width=True)

            # Itemized Component Table
            st.subheader("Component Placement Analysis Table")
            c_rows = []
            for cr in raw_res.component_results:
                c_rows.append({
                    "Component": cr.component_name,
                    "Type": cr.component_type.upper(),
                    "Status": cr.status,
                    "Shift (dx, dy)": f"({cr.shift_dx:.1f}, {cr.shift_dy:.1f}) px",
                    "Total Shift": f"{cr.shift_distance:.1f} px",
                    "Rotation Dev": f"{cr.rotation_deviation_deg:.1f}°",
                    "NCC Score": f"{cr.match_score:.3f}",
                    "Edge Ratio": f"{cr.edge_ratio:.2f}",
                    "Diagnostic Reason": cr.anomaly_details
                })
            df_comp = pd.DataFrame(c_rows)
            # Put anomalies first
            df_comp["is_anomaly"] = df_comp["Status"] != "OK"
            df_comp = df_comp.sort_values(by="is_anomaly", ascending=False).drop(columns=["is_anomaly"])
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            # Export Reports
            st.markdown("---")
            st.subheader("Export Inspection Audit Reports")
            e1, e2, e3 = st.columns(3)
            with e1:
                json_p = Path("outputs/reports") / f"{verdict.inspection_id}.json"
                JSONReporter.save_report(verdict, json_p)
                with open(json_p, "r", encoding="utf-8") as f:
                    st.download_button("Download JSON Report", f.read(), file_name=f"{verdict.inspection_id}.json", mime="application/json")
            with e2:
                csv_p = Path("outputs/reports") / f"{verdict.inspection_id}_defects.csv"
                CSVReporter.export_defects_csv(verdict, csv_p)
                with open(csv_p, "r", encoding="utf-8") as f:
                    st.download_button("Download CSV Placement Log", f.read(), file_name=f"{verdict.inspection_id}.csv", mime="text/csv")
            with e3:
                pdf_p = Path("outputs/reports") / f"{verdict.inspection_id}.pdf"
                PDFReporter.generate_pdf_report(verdict, pdf_p)
                with open(pdf_p, "rb") as f:
                    st.download_button("Download PDF Audit Report", f.read(), file_name=f"{verdict.inspection_id}.pdf", mime="application/pdf")
