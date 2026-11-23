"""Unified Industrial AOI Inspection Decision Engine."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from app.config import DecisionEngineConfig
from app.inspection.bare_pcb.pipeline import BarePCBInspectionResult
from app.inspection.assembled_pcb.pipeline import PCBAInspectionResult


@dataclass
class StandardizedDefectItem:
    """Normalized defect record common across Bare PCB and PCBA domains."""
    id: int
    category: str              # e.g. "open", "short", "missing_component", "shifted_component"
    severity: str              # "CRITICAL", "MAJOR", "MINOR"
    bounding_box: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: float
    confidence: float
    rule_applied: str
    details: str


@dataclass
class StandardizedInspectionVerdict:
    """Unified verdict container ready for UI rendering, audit logging, and MES reporting."""
    inspection_id: str
    timestamp: str
    inspection_type: str       # "Bare PCB Fabrication" or "Assembled PCB (PCBA)"
    decision: str              # "PASS", "FAIL", "REVIEW"
    decision_reason: str
    total_defect_count: int
    defect_summary: Dict[str, int] = field(default_factory=dict)
    defects: List[StandardizedDefectItem] = field(default_factory=list)
    processing_time_ms: float = 0.0
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    algorithm_info: Dict[str, Any] = field(default_factory=dict)
    visualizations: Dict[str, np.ndarray] = field(default_factory=dict)


class UnifiedDecisionEngine:
    """Industrial decision arbiter enforcing safety, confidence, and quality criteria."""

    CRITICAL_DEFECTS = {"open", "short", "MISSING", "missing_component"}

    def __init__(self, config: Optional[DecisionEngineConfig] = None):
        self.config = config or DecisionEngineConfig()

    def arbitrate_bare_pcb(
        self,
        raw_result: BarePCBInspectionResult,
        input_image_id: str = "bare_pcb"
    ) -> StandardizedInspectionVerdict:
        """Evaluate Bare PCB inspection output against industrial manufacturing criteria."""
        standardized_defects: List[StandardizedDefectItem] = []
        reg_info = raw_result.registration_info
        inlier_ratio = float(reg_info.get("inlier_ratio", 1.0))
        align_error = float(reg_info.get("error_px", 0.0))

        # 1. Quality & Confidence Pre-Check
        if not reg_info.get("is_aligned", True):
            decision = "REVIEW"
            reason = f"Image registration was uncertain: {reg_info.get('message', 'Poor alignment')}"
        elif inlier_ratio < self.config.min_registration_inlier_ratio:
            decision = "REVIEW"
            reason = f"Registration inlier ratio {inlier_ratio:.2f} is below minimum acceptable threshold {self.config.min_registration_inlier_ratio:.2f}."
        elif align_error > self.config.max_alignment_error_px:
            decision = "REVIEW"
            reason = f"Alignment residual error {align_error:.1f}px exceeds tolerance limit of {self.config.max_alignment_error_px}px."
        else:
            # 2. Defect Severity & Classification
            for d in raw_result.defects:
                severity = "CRITICAL" if d.class_name in self.CRITICAL_DEFECTS else "MAJOR"
                standardized_defects.append(
                    StandardizedDefectItem(
                        id=d.defect_id,
                        category=d.class_name,
                        severity=severity,
                        bounding_box=d.bbox,
                        centroid=d.centroid,
                        area=d.area,
                        confidence=d.confidence,
                        rule_applied=d.rule_applied,
                        details=f"Polarity: {d.polarity}, Area: {d.area}px, Conf: {d.confidence:.2f}"
                    )
                )

            if len(standardized_defects) > 0:
                decision = "FAIL"
                crit_count = sum(1 for d in standardized_defects if d.severity == "CRITICAL")
                reason = f"Detected {len(standardized_defects)} manufacturing defect(s) ({crit_count} critical: open/short)."
            else:
                decision = "PASS"
                reason = "All bare copper traces and dielectric spaces conform to reference template."

        return StandardizedInspectionVerdict(
            inspection_id=raw_result.inspection_id,
            timestamp=raw_result.timestamp,
            inspection_type="Bare PCB Fabrication",
            decision=decision,
            decision_reason=reason,
            total_defect_count=len(standardized_defects),
            defect_summary=raw_result.defect_counts_by_type,
            defects=standardized_defects,
            processing_time_ms=raw_result.processing_time_ms,
            quality_metrics={
                "registration_inliers": inlier_ratio,
                "alignment_error_px": align_error,
                "subtractive_pixel_diff": raw_result.binary_defect_mask.sum() / 255 if raw_result.binary_defect_mask is not None else 0
            },
            algorithm_info={
                "pipeline": "BarePCBInspectionPipeline",
                "method": "Classical Dual-Polarity Subtraction + Connected Components + Morphology",
                "rules": "OpenCV Morphological & Topological Rules (DeepPCB Spec)"
            },
            visualizations={
                "original_reference": raw_result.original_reference,
                "original_test": raw_result.original_test,
                "aligned_test": raw_result.aligned_test,
                "difference_image": raw_result.difference_image,
                "binary_mask": raw_result.binary_defect_mask,
                "contour_vis": raw_result.contour_visualization,
                "bbox_vis": raw_result.bbox_visualization
            }
        )

    def arbitrate_pcba(
        self,
        raw_result: PCBAInspectionResult,
        input_image_id: str = "pcba"
    ) -> StandardizedInspectionVerdict:
        """Evaluate Assembled PCBA inspection output against industrial placement criteria."""
        standardized_defects: List[StandardizedDefectItem] = []
        reg_info = raw_result.registration_info
        inlier_ratio = float(reg_info.get("inlier_ratio", 1.0))
        align_error = float(reg_info.get("error_px", 0.0))

        # 1. Quality & Confidence Pre-Check
        if not reg_info.get("is_aligned", True):
            decision = "REVIEW"
            reason = f"Macro board alignment uncertain: {reg_info.get('message', 'Alignment issue')}"
        elif inlier_ratio < self.config.min_registration_inlier_ratio:
            decision = "REVIEW"
            reason = f"Registration inlier ratio {inlier_ratio:.2f} is below minimum threshold."
        else:
            # 2. Extract Anomalies
            for idx, c in enumerate(raw_result.component_results, 1):
                if c.status == "OK":
                    continue
                severity = "CRITICAL" if c.status == "MISSING" else ("MAJOR" if c.status == "ROTATED" else "MINOR")
                standardized_defects.append(
                    StandardizedDefectItem(
                        id=idx,
                        category=f"{c.status.lower()}_component",
                        severity=severity,
                        bounding_box=c.detected_bbox if c.status != "MISSING" else c.nominal_bbox,
                        centroid=((c.nominal_bbox[0]+c.nominal_bbox[2])/2.0, (c.nominal_bbox[1]+c.nominal_bbox[3])/2.0),
                        area=float((c.nominal_bbox[2]-c.nominal_bbox[0]) * (c.nominal_bbox[3]-c.nominal_bbox[1])),
                        confidence=c.confidence,
                        rule_applied=f"SMT Placement Check: {c.status}",
                        details=c.anomaly_details
                    )
                )

            if raw_result.anomaly_counts_by_type.get("REVIEW", 0) > (0.2 * max(1, raw_result.total_components_inspected)):
                decision = "REVIEW"
                reason = "Over 20% of component regions require manual engineering review."
            elif len(standardized_defects) > 0:
                decision = "FAIL"
                crit = sum(1 for d in standardized_defects if d.severity == "CRITICAL")
                reason = f"Found {len(standardized_defects)} placement anomalie(s) ({crit} missing parts)."
            else:
                decision = "PASS"
                reason = f"All {raw_result.total_components_inspected} components verified present and within nominal SMT tolerances."

        return StandardizedInspectionVerdict(
            inspection_id=raw_result.inspection_id,
            timestamp=raw_result.timestamp,
            inspection_type="Assembled PCB (PCBA)",
            decision=decision,
            decision_reason=reason,
            total_defect_count=len(standardized_defects),
            defect_summary=raw_result.anomaly_counts_by_type,
            defects=standardized_defects,
            processing_time_ms=raw_result.processing_time_ms,
            quality_metrics={
                "registration_inliers": inlier_ratio,
                "alignment_error_px": align_error,
                "total_components": raw_result.total_components_inspected
            },
            algorithm_info={
                "pipeline": "PCBAInspectionPipeline",
                "method": "Board Homography + Normalized Cross-Correlation + Moment Orientation",
                "rules": "SMT Component Placement Spec (Missing, Shift, Rotation)"
            },
            visualizations={
                "original_reference": raw_result.original_reference,
                "original_test": raw_result.original_test,
                "aligned_test": raw_result.aligned_test,
                "difference_image": raw_result.difference_image,
                "component_regions": raw_result.component_region_vis,
                "bbox_vis": raw_result.bbox_visualization
            }
        )
