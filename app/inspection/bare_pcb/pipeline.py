"""End-to-End Bare PCB Fabrication Inspection Pipeline."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.preprocessing.pipeline import PreprocessingPipeline
from app.inspection.registration import ImageRegistrar, RegistrationResult
from .diff_analyzer import DifferenceAnalyzer, DifferenceResult
from .defect_extractor import DefectExtractor, DefectCandidate
from .classifier import ClassicalDefectClassifier, ClassifiedDefect


@dataclass
class BarePCBInspectionResult:
    """Full structured inspection result for bare PCB fabrication."""
    inspection_id: str
    timestamp: str
    inspection_type: str = "Bare PCB Fabrication"
    decision: str = "PASS"  # PASS, FAIL, REVIEW
    total_defects: int = 0
    defect_counts_by_type: Dict[str, int] = field(default_factory=dict)
    defects: List[ClassifiedDefect] = field(default_factory=list)
    processing_time_ms: float = 0.0
    registration_info: Dict[str, Any] = field(default_factory=dict)
    
    # Visualizations (NumPy BGR images for inspection display)
    original_reference: Optional[np.ndarray] = None
    original_test: Optional[np.ndarray] = None
    aligned_test: Optional[np.ndarray] = None
    difference_image: Optional[np.ndarray] = None
    binary_defect_mask: Optional[np.ndarray] = None
    contour_visualization: Optional[np.ndarray] = None
    bbox_visualization: Optional[np.ndarray] = None
    intermediate_stages: Dict[str, np.ndarray] = field(default_factory=dict)
    config_applied: Dict[str, Any] = field(default_factory=dict)


class BarePCBInspectionPipeline:
    """Production classical CV pipeline for Bare PCB Automated Optical Inspection."""

    # Color palette for defect classes (BGR)
    DEFECT_COLORS = {
        "open": (0, 0, 255),        # Red
        "short": (0, 165, 255),     # Orange
        "mousebite": (255, 0, 255), # Magenta
        "spur": (255, 255, 0),      # Cyan
        "copper": (0, 255, 255),    # Yellow
        "pin-hole": (255, 0, 0),    # Blue
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        prep_cfg = self.config.get("preprocessing", {})
        reg_cfg = self.config.get("registration", {})
        diff_cfg = self.config.get("defect_extraction", {})

        self.preprocessor = PreprocessingPipeline(config=prep_cfg)
        self.registrar = ImageRegistrar(
            max_features=int(reg_cfg.get("max_features", 1000)),
            match_ratio=float(reg_cfg.get("match_ratio", 0.75)),
            ransac_threshold=float(reg_cfg.get("ransac_reproj_threshold", 3.0)),
            transform_type=reg_cfg.get("transform_type", "affine")
        )
        self.diff_analyzer = DifferenceAnalyzer(
            morph_kernel_size=int(diff_cfg.get("morph_kernel_size", 3)),
            morph_close_kernel_size=int(diff_cfg.get("morph_close_kernel_size", 7)),
            morph_iterations=int(diff_cfg.get("morph_iterations", 1))
        )
        self.defect_extractor = DefectExtractor(
            min_area=int(diff_cfg.get("min_area", 25)),
            max_area=int(diff_cfg.get("max_area", 12000)),
            bbox_padding=int(diff_cfg.get("bbox_padding", 8))
        )
        self.classifier = ClassicalDefectClassifier()

    def inspect(
        self,
        reference_bgr: np.ndarray,
        test_bgr: np.ndarray,
        inspection_id: Optional[str] = None
    ) -> BarePCBInspectionResult:
        """Execute the end-to-end bare PCB inspection."""
        t_start = time.perf_counter()
        iid = inspection_id or f"INSP-BARE-{int(time.time()*1000)}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # Step 1: Preprocess reference and test images
        prep_ref = self.preprocessor.process_bare_pcb(reference_bgr)
        prep_test = self.preprocessor.process_bare_pcb(test_bgr)

        ref_bin = prep_ref.final_image
        test_bin = prep_test.final_image

        # Step 2: Image Registration and Alignment
        reg_res = self.registrar.register(
            template_gray=ref_bin,
            test_gray=test_bin,
            test_color=test_bgr
        )
        aligned_test_bgr = reg_res.aligned_image
        # Re-binarize aligned test image
        _, aligned_test_bin = cv2.threshold(
            cv2.cvtColor(aligned_test_bgr, cv2.COLOR_BGR2GRAY) if len(aligned_test_bgr.shape) == 3 else aligned_test_bgr,
            127, 255, cv2.THRESH_BINARY
        )

        # Check for alignment failure triggering REVIEW
        is_alignment_ok = reg_res.is_aligned
        reg_info = {
            "is_aligned": reg_res.is_aligned,
            "method": reg_res.method,
            "inlier_ratio": reg_res.inlier_ratio,
            "error_px": reg_res.alignment_error_px,
            "message": reg_res.message
        }

        # Step 3: Dual-Polarity Difference Analysis
        diff_res = self.diff_analyzer.analyze(
            template_binary=ref_bin,
            aligned_test_binary=aligned_test_bin
        )

        # Step 4: Defect Extraction
        deficit_candidates = self.defect_extractor.extract_candidates(
            difference_mask=diff_res.copper_deficit_clean,
            polarity="deficit",
            template_binary=ref_bin
        )
        excess_candidates = self.defect_extractor.extract_candidates(
            difference_mask=diff_res.copper_excess_clean,
            polarity="excess",
            template_binary=ref_bin
        )
        all_candidates = deficit_candidates + excess_candidates

        # Step 5: Classical Rule-Based Classification
        classified_defects: List[ClassifiedDefect] = []
        counts: Dict[str, int] = {
            "open": 0, "short": 0, "mousebite": 0, "spur": 0, "copper": 0, "pin-hole": 0
        }
        for cand in all_candidates:
            c_defect = self.classifier.classify(cand)
            classified_defects.append(c_defect)
            counts[c_defect.class_name] = counts.get(c_defect.class_name, 0) + 1

        # Step 6: Visualizations Generation
        # Difference Heatmap (Deficit in Blue, Excess in Red)
        h, w = ref_bin.shape[:2]
        diff_vis = np.zeros((h, w, 3), dtype=np.uint8)
        diff_vis[:, :, 0] = diff_res.copper_deficit_clean  # Blue channel = missing copper
        diff_vis[:, :, 2] = diff_res.copper_excess_clean   # Red channel = extra copper
        # Where both, make magenta
        diff_vis[diff_res.combined_clean > 0, 1] = 40

        # Contour Visualization
        contour_vis = aligned_test_bgr.copy()
        for defect in classified_defects:
            col = self.DEFECT_COLORS.get(defect.class_name, (0, 255, 0))
            for cand in all_candidates:
                if cand.id == defect.defect_id and cand.polarity == defect.polarity:
                    cv2.drawContours(contour_vis, [cand.contour], -1, col, 2)
                    break

        # Bounding Box Visualization
        bbox_vis = aligned_test_bgr.copy()
        for idx, defect in enumerate(classified_defects, 1):
            col = self.DEFECT_COLORS.get(defect.class_name, (0, 255, 0))
            x1, y1, x2, y2 = defect.bbox
            cv2.rectangle(bbox_vis, (x1, y1), (x2, y2), col, 2)
            label = f"{defect.class_name} ({defect.confidence:.2f})"
            # Draw label banner
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(bbox_vis, (x1, max(0, y1 - lh - 6)), (x1 + lw + 4, y1), col, -1)
            cv2.putText(bbox_vis, label, (x1 + 2, max(lh + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Step 7: Decision Evaluation
        if not is_alignment_ok and reg_res.num_matches > 0:
            decision = "REVIEW"
        elif len(classified_defects) > 0:
            decision = "FAIL"
        else:
            decision = "PASS"

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Step 8: Save intermediate and visualization images to disk
        out_dir = Path("outputs/intermediate") / iid
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "01_ref_binary.png"), ref_bin)
        cv2.imwrite(str(out_dir / "02_test_binary.png"), test_bin)
        cv2.imwrite(str(out_dir / "03_aligned_test.png"), aligned_test_bgr)
        cv2.imwrite(str(out_dir / "04_copper_deficit.png"), diff_res.copper_deficit_clean)
        cv2.imwrite(str(out_dir / "05_copper_excess.png"), diff_res.copper_excess_clean)
        cv2.imwrite(str(out_dir / "06_combined_diff.png"), diff_res.combined_clean)
        cv2.imwrite(str(out_dir / "07_difference_image.png"), diff_vis)
        cv2.imwrite(str(out_dir / "08_contour_vis.png"), contour_vis)
        cv2.imwrite(str(out_dir / "09_bbox_vis.png"), bbox_vis)

        return BarePCBInspectionResult(
            inspection_id=iid,
            timestamp=timestamp,
            inspection_type="Bare PCB Fabrication",
            decision=decision,
            total_defects=len(classified_defects),
            defect_counts_by_type=counts,
            defects=classified_defects,
            processing_time_ms=t_elapsed_ms,
            registration_info=reg_info,
            original_reference=reference_bgr,
            original_test=test_bgr,
            aligned_test=aligned_test_bgr,
            difference_image=diff_vis,
            binary_defect_mask=diff_res.combined_clean,
            contour_visualization=contour_vis,
            bbox_visualization=bbox_vis,
            intermediate_stages={
                "ref_binary": ref_bin,
                "test_binary": test_bin,
                "copper_deficit": diff_res.copper_deficit_clean,
                "copper_excess": diff_res.copper_excess_clean,
                "combined_diff": diff_res.combined_clean
            },
            config_applied=self.config
        )
