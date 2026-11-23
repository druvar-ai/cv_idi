"""Page 2: Interactive Bare PCB Fabrication Inspection."""

import glob
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app.acquisition.loader import ImageLoader
from app.config import get_config
from app.inspection.bare_pcb.pipeline import BarePCBInspectionPipeline
from app.inspection.decision_engine import UnifiedDecisionEngine
from app.reporting.json_reporter import JSONReporter
from app.reporting.csv_reporter import CSVReporter
from app.reporting.pdf_reporter import PDFReporter
from app.storage.history_manager import HistoryManager
from app.ui.components import get_status_badge_html


def render_bare_pcb_page(history: HistoryManager) -> None:
    """Render interactive bare PCB defect detection and visual difference inspector."""
    st.markdown('<div class="main-header">Bare PCB Fabrication Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Module A: Detection of Open Circuits, Shorts, Mouse Bites, Spurs, Pin Holes & Spurious Copper</div>', unsafe_allow_html=True)

    cfg = get_config()
    pipeline = BarePCBInspectionPipeline(config=cfg.bare_pcb)
    engine = UnifiedDecisionEngine(config=cfg.decision_engine)

    # Input Mode Selection
    mode = st.radio("Select Image Acquisition Source:", ["Curated DeepPCB Benchmark Samples", "Upload Custom Image Pair"], horizontal=True)

    ref_bgr = None
    test_bgr = None
    sample_id = "custom"

    if mode == "Curated DeepPCB Benchmark Samples":
        sample_dir = Path("datasets/bare_pcb")
        img_files = sorted(glob.glob(str(sample_dir / "images" / "*_test.jpg")))
        if not img_files:
            st.error("No curated bare PCB samples found in datasets/bare_pcb/images.")
            return

        sample_names = [Path(p).stem.replace("_test", "") for p in img_files]
        col_sel, col_mode = st.columns([2, 1])
        with col_sel:
            selected_sample = st.selectbox("Select Test Board:", sample_names, index=0)
        with col_mode:
            test_type = st.radio("Test Target:", ["Defective Board", "Defect-Free Reference (Pass Test)"], horizontal=True)

        sample_id = selected_sample
        ref_path = sample_dir / "templates" / f"{selected_sample}_temp.jpg"
        if test_type == "Defective Board":
            test_path = sample_dir / "images" / f"{selected_sample}_test.jpg"
        else:
            test_path = ref_path

        ref_img_obj, val_ref = ImageLoader.load_from_path(ref_path)
        test_img_obj, val_test = ImageLoader.load_from_path(test_path)

        if ref_img_obj and test_img_obj:
            ref_bgr = ref_img_obj.image_bgr
            test_bgr = test_img_obj.image_bgr
    else:
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            u_ref = st.file_uploader("Upload Golden Reference (Template) PCB", type=["jpg", "jpeg", "png", "bmp"])
        with u_col2:
            u_test = st.file_uploader("Upload Test PCB Board", type=["jpg", "jpeg", "png", "bmp"])

        if u_ref and u_test:
            ref_img_obj, val_ref = ImageLoader.load_from_bytes(u_ref.read(), source_name=u_ref.name)
            test_img_obj, val_test = ImageLoader.load_from_bytes(u_test.read(), source_name=u_test.name)
            if not val_ref.is_valid:
                st.error(f"Reference Image Error: {val_ref.message}")
            elif not val_test.is_valid:
                st.error(f"Test Image Error: {val_test.message}")
            else:
                ref_bgr = ref_img_obj.image_bgr
                test_bgr = test_img_obj.image_bgr
                sample_id = Path(u_test.name).stem

    if ref_bgr is not None and test_bgr is not None:
        # Pre-flight check preview
        st.success(f"Inputs Validated: Reference ({ref_bgr.shape[1]}x{ref_bgr.shape[0]}), Test ({test_bgr.shape[1]}x{test_bgr.shape[0]})")

        if st.button("Start Classical AOI Inspection", type="primary", use_container_width=True):
            with st.spinner("Executing Classical Computer Vision Pipeline..."):
                raw_res = pipeline.inspect(reference_bgr=ref_bgr, test_bgr=test_bgr, inspection_id=f"INSP-BARE-{sample_id}")
                verdict = engine.arbitrate_bare_pcb(raw_res)
                history.add_inspection(verdict)

            # Display Verdict Summary Banner
            st.markdown("---")
            v_col1, v_col2, v_col3, v_col4 = st.columns([1.5, 2.5, 1.5, 1.5])
            with v_col1:
                st.markdown(f"### Verdict: {get_status_badge_html(verdict.decision)}", unsafe_allow_html=True)
            with v_col2:
                st.markdown(f"**Decision Reason:** {verdict.decision_reason}")
            with v_col3:
                st.metric("Defects Detected", verdict.total_defect_count)
            with v_col4:
                st.metric("Processing Latency", f"{verdict.processing_time_ms:.1f} ms")

            # Multi-Tab Visual Stage Breakdown
            tabs = st.tabs([
                "1. Detected Bounding Boxes",
                "2. Dual-Polarity Difference Map",
                "3. Contours & Defect Mask",
                "4. Alignment & Registration",
                "5. Side-by-Side Original Inputs",
                "6. Intermediate Pipeline Stages"
            ])

            with tabs[0]:
                st.image(cv2.cvtColor(verdict.visualizations["bbox_vis"], cv2.COLOR_BGR2RGB), caption="Defect Bounding Box Overlays (DeepPCB Rules)", use_container_width=True)

            with tabs[1]:
                c_diff1, c_diff2 = st.columns(2)
                with c_diff1:
                    st.image(cv2.cvtColor(verdict.visualizations["difference_image"], cv2.COLOR_BGR2RGB), caption="Dual-Polarity Map (Blue=Copper Deficit, Red=Copper Excess)", use_container_width=True)
                with c_diff2:
                    st.image(verdict.visualizations["binary_mask"], caption="Cleaned Binary Defect Mask", use_container_width=True)

            with tabs[2]:
                st.image(cv2.cvtColor(verdict.visualizations["contour_vis"], cv2.COLOR_BGR2RGB), caption="Defect Contours & Morphological Outlines", use_container_width=True)

            with tabs[3]:
                c_al1, c_al2 = st.columns(2)
                with c_al1:
                    st.image(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB), caption="Reference Coordinate Space", use_container_width=True)
                with c_al2:
                    st.image(cv2.cvtColor(verdict.visualizations["aligned_test"], cv2.COLOR_BGR2RGB), caption=f"Aligned Test Image (Residual: {verdict.quality_metrics.get('alignment_error_px', 0):.2f}px)", use_container_width=True)

            with tabs[4]:
                c_in1, c_in2 = st.columns(2)
                with c_in1:
                    st.image(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB), caption="Original Template Image", use_container_width=True)
                with c_in2:
                    st.image(cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB), caption="Original Test Image", use_container_width=True)

            with tabs[5]:
                st.write("Inspecting internal mathematical filters and thresholding:")
                st_cols = st.columns(3)
                stages = raw_res.intermediate_stages
                if "ref_binary" in stages:
                    with st_cols[0]:
                        st.image(stages["ref_binary"], caption="Stage 1: Reference Otsu Mask", use_container_width=True)
                if "test_binary" in stages:
                    with st_cols[1]:
                        st.image(stages["test_binary"], caption="Stage 2: Test Otsu Mask", use_container_width=True)
                if "combined_diff" in stages:
                    with st_cols[2]:
                        st.image(stages["combined_diff"], caption="Stage 3: Subtraction Map", use_container_width=True)

            # Granular Defect Items Table
            if verdict.defects:
                st.subheader("Itemized Defect Register")
                d_rows = []
                for d in verdict.defects:
                    d_rows.append({
                        "ID": d.id,
                        "Category": d.category.upper(),
                        "Severity": d.severity,
                        "Bounding Box [x1, y1, x2, y2]": str(list(d.bounding_box)),
                        "Area (px)": f"{d.area:.0f}",
                        "Confidence": f"{d.confidence:.2f}",
                        "Classification Rule Applied": d.rule_applied
                    })
                st.dataframe(pd.DataFrame(d_rows), use_container_width=True, hide_index=True)

            # Export Buttons
            st.markdown("---")
            st.subheader("Export Inspection Audit Reports")
            e1, e2, e3 = st.columns(3)
            with e1:
                json_path = Path("outputs/reports") / f"{verdict.inspection_id}.json"
                JSONReporter.save_report(verdict, json_path)
                with open(json_path, "r", encoding="utf-8") as f:
                    st.download_button("Download JSON Report", f.read(), file_name=f"{verdict.inspection_id}.json", mime="application/json")
            with e2:
                csv_path = Path("outputs/reports") / f"{verdict.inspection_id}_defects.csv"
                CSVReporter.export_defects_csv(verdict, csv_path)
                with open(csv_path, "r", encoding="utf-8") as f:
                    st.download_button("Download CSV Defects", f.read(), file_name=f"{verdict.inspection_id}.csv", mime="text/csv")
            with e3:
                pdf_path = Path("outputs/reports") / f"{verdict.inspection_id}.pdf"
                PDFReporter.generate_pdf_report(verdict, pdf_path)
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF Audit Report", f.read(), file_name=f"{verdict.inspection_id}.pdf", mime="application/pdf")
