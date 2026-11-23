"""End-to-End Assembled PCB (PCBA) Classical CV Inspection Pipeline."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from app.preprocessing.pipeline import PreprocessingPipeline
from app.inspection.registration import ImageRegistrar, RegistrationResult
from .component_detector import ComponentROI, ComponentSpecificationLoader
from .anomaly_analyzer import ComponentAnomalyAnalyzer, ComponentInspectionResult


@dataclass
class PCBAInspectionResult:
    """Full structured inspection verdict and intermediate outputs for assembled PCBA."""
    inspection_id: str
    timestamp: str
    inspection_type: str = "Assembled PCB (PCBA)"
    decision: str = "PASS"  # PASS, FAIL, REVIEW
    total_components_inspected: int = 0
    total_anomalies: int = 0
    anomaly_counts_by_type: Dict[str, int] = field(default_factory=dict)
    component_results: List[ComponentInspectionResult] = field(default_factory=list)
    processing_time_ms: float = 0.0
    registration_info: Dict[str, Any] = field(default_factory=dict)

    # Inspectable visualizations (NumPy BGR images)
    original_reference: Optional[np.ndarray] = None
    original_test: Optional[np.ndarray] = None
    aligned_test: Optional[np.ndarray] = None
    difference_image: Optional[np.ndarray] = None
    component_region_vis: Optional[np.ndarray] = None
    bbox_visualization: Optional[np.ndarray] = None
    intermediate_stages: Dict[str, np.ndarray] = field(default_factory=dict)
    config_applied: Dict[str, Any] = field(default_factory=dict)


class PCBAInspectionPipeline:
    """Industrial classical computer vision pipeline for PCBA SMT inspection."""

    STATUS_COLORS = {
        "OK": (0, 200, 0),        # Green
        "MISSING": (0, 0, 255),   # Red
        "SHIFTED": (0, 165, 255), # Orange
        "ROTATED": (255, 0, 255), # Magenta
        "REVIEW": (0, 255, 255),  # Yellow
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        prep_cfg = self.config.get("preprocessing", {})
        reg_cfg = self.config.get("registration", {})
        comp_cfg = self.config.get("component_inspection", {})

        self.resize_factor = float(prep_cfg.get("resize_factor", 0.5))
        self.preprocessor = PreprocessingPipeline(config=prep_cfg)
        self.registrar = ImageRegistrar(
            max_features=int(reg_cfg.get("max_features", 2500)),
            match_ratio=float(reg_cfg.get("match_ratio", 0.70)),
            ransac_threshold=float(reg_cfg.get("ransac_reproj_threshold", 4.0)),
            transform_type=reg_cfg.get("transform_type", "homography")
        )
        self.analyzer = ComponentAnomalyAnalyzer(
            missing_correlation_threshold=float(comp_cfg.get("missing_correlation_threshold", 0.45)),
            missing_edge_density_ratio=float(comp_cfg.get("missing_edge_density_ratio", 0.40)),
            max_allowed_shift_px=float(comp_cfg.get("max_allowed_shift_px", 14.0)),
            rotation_tolerance_deg=float(comp_cfg.get("rotation_tolerance_deg", 12.0)),
            search_margin_px=int(comp_cfg.get("search_margin_px", 24))
        )

    def inspect(
        self,
        reference_bgr: np.ndarray,
        test_bgr: np.ndarray,
        components_spec: Optional[Union[List[ComponentROI], str, Path]] = None,
        inspection_id: Optional[str] = None
    ) -> PCBAInspectionResult:
        """Execute PCBA inspection comparing test board against reference template."""
        t_start = time.perf_counter()
        iid = inspection_id or f"INSP-PCBA-{int(time.time()*1000)}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # Step 1: Preprocess reference and test boards
        prep_ref = self.preprocessor.process_pcba(reference_bgr, resize_factor=self.resize_factor)
        prep_test = self.preprocessor.process_pcba(test_bgr, resize_factor=self.resize_factor)

        ref_proc = prep_ref.final_image
        test_proc = prep_test.final_image
        ref_resized_bgr = prep_ref.stages["1_resized"]
        test_resized_bgr = prep_test.stages["1_resized"]

        # Step 2: Board-Level Alignment
        reg_res = self.registrar.register(
            template_gray=ref_proc,
            test_gray=test_proc,
            test_color=ref_resized_bgr  # Placeholder, we will warp test
        )
        # Warp test image (both color display and preprocessed gray) using the estimated matrix
        if reg_res.transform_matrix is not None:
            h_ref, w_ref = ref_proc.shape[:2]
            if reg_res.transform_matrix.shape[0] == 2:
                aligned_test_bgr = cv2.warpAffine(test_resized_bgr, reg_res.transform_matrix, (w_ref, h_ref), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                aligned_test_gray = cv2.warpAffine(test_proc, reg_res.transform_matrix, (w_ref, h_ref), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            else:
                aligned_test_bgr = cv2.warpPerspective(test_resized_bgr, reg_res.transform_matrix, (w_ref, h_ref), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                aligned_test_gray = cv2.warpPerspective(test_proc, reg_res.transform_matrix, (w_ref, h_ref), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            aligned_test_bgr = test_resized_bgr.copy()
            aligned_test_gray = test_proc.copy()

        reg_info = {
            "is_aligned": reg_res.is_aligned,
            "method": reg_res.method,
            "inlier_ratio": reg_res.inlier_ratio,
            "error_px": reg_res.alignment_error_px,
            "message": reg_res.message
        }

        # Step 3: Resolve Component ROIs
        # Load from spec (XML / List) or auto-segment
        rois: List[ComponentROI] = []
        if isinstance(components_spec, (str, Path)):
            raw_rois = ComponentSpecificationLoader.load_from_xml(components_spec)
            # Scale coordinates according to resize_factor
            for r in raw_rois:
                sf = self.resize_factor
                scaled_box = (
                    int(r.bbox[0] * sf),
                    int(r.bbox[1] * sf),
                    int(r.bbox[2] * sf),
                    int(r.bbox[3] * sf)
                )
                rois.append(ComponentROI(
                    name=r.name,
                    component_type=r.component_type,
                    bbox=scaled_box,
                    nominal_center=(scaled_box[0] + (scaled_box[2]-scaled_box[0])/2.0, scaled_box[1] + (scaled_box[3]-scaled_box[1])/2.0),
                    width=max(1, scaled_box[2] - scaled_box[0]),
                    height=max(1, scaled_box[3] - scaled_box[1])
                ))
        elif isinstance(components_spec, list):
            rois = components_spec
        else:
            # Auto-segment components
            rois = ComponentSpecificationLoader.auto_segment_components(ref_proc)

        # Step 4: Component Inspection and Anomaly Analysis
        comp_results: List[ComponentInspectionResult] = []
        counts: Dict[str, int] = {"OK": 0, "MISSING": 0, "SHIFTED": 0, "ROTATED": 0, "REVIEW": 0}

        for comp in rois:
            res = self.analyzer.analyze_component(
                component=comp,
                template_gray=ref_proc,
                aligned_test_gray=aligned_test_gray
            )
            comp_results.append(res)
            counts[res.status] = counts.get(res.status, 0) + 1

        total_anomalies = counts["MISSING"] + counts["SHIFTED"] + counts["ROTATED"]

        # Step 5: Visualizations Generation
        # Photometric Difference Image
        diff_raw = cv2.absdiff(ref_proc, aligned_test_gray)
        diff_vis = cv2.applyColorMap(diff_raw, cv2.COLORMAP_JET)

        # Component Regions Grid Overlay
        comp_grid_vis = ref_resized_bgr.copy()
        for c in rois:
            x1, y1, x2, y2 = c.bbox
            cv2.rectangle(comp_grid_vis, (x1, y1), (x2, y2), (255, 200, 0), 1)

        # Bounding Box Overlays with Industrial Status Badges
        bbox_vis = aligned_test_bgr.copy()
        for cr in comp_results:
            col = self.STATUS_COLORS.get(cr.status, (255, 255, 255))
            x1, y1, x2, y2 = cr.detected_bbox if cr.status != "MISSING" else cr.nominal_bbox
            thickness = 2 if cr.status == "OK" else 3
            cv2.rectangle(bbox_vis, (x1, y1), (x2, y2), col, thickness)

            # Draw label only for anomalies or a sample of OKs
            if cr.status != "OK":
                label = f"{cr.component_name}: {cr.status} ({cr.confidence:.2f})"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(bbox_vis, (x1, max(0, y1 - lh - 6)), (x1 + lw + 4, y1), col, -1)
                cv2.putText(
                    bbox_vis, label, (x1 + 2, max(lh + 2, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )

        # Step 6: Industrial AOI Decision Engine
        if not reg_res.is_aligned:
            decision = "REVIEW"
        elif counts["REVIEW"] > (0.2 * max(1, len(rois))):
            decision = "REVIEW"
        elif total_anomalies > 0:
            decision = "FAIL"
        else:
            decision = "PASS"

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Step 7: Save intermediate and visualization images to disk
        out_dir = Path("outputs/intermediate") / iid
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "01_ref_preprocessed.png"), ref_proc)
        cv2.imwrite(str(out_dir / "02_test_preprocessed.png"), test_proc)
        cv2.imwrite(str(out_dir / "03_aligned_test.png"), aligned_test_bgr)
        cv2.imwrite(str(out_dir / "04_abs_diff.png"), diff_raw)
        cv2.imwrite(str(out_dir / "05_diff_heatmap.png"), diff_vis)
        cv2.imwrite(str(out_dir / "06_component_regions.png"), comp_grid_vis)
        cv2.imwrite(str(out_dir / "07_bbox_vis.png"), bbox_vis)

        return PCBAInspectionResult(
            inspection_id=iid,
            timestamp=timestamp,
            inspection_type="Assembled PCB (PCBA)",
            decision=decision,
            total_components_inspected=len(rois),
            total_anomalies=total_anomalies,
            anomaly_counts_by_type=counts,
            component_results=comp_results,
            processing_time_ms=t_elapsed_ms,
            registration_info=reg_info,
            original_reference=reference_bgr,
            original_test=test_bgr,
            aligned_test=aligned_test_bgr,
            difference_image=diff_vis,
            component_region_vis=comp_grid_vis,
            bbox_visualization=bbox_vis,
            intermediate_stages={
                "ref_edges": prep_ref.stages["5_edges"],
                "test_edges": prep_test.stages["5_edges"],
                "abs_diff": diff_raw,
            },
            config_applied=self.config
        )
