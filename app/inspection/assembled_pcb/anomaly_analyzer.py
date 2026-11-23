"""Component Anomaly Analyzer for Missing, Shifted, and Rotated Components."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .component_detector import ComponentROI


@dataclass
class ComponentInspectionResult:
    """Detailed inspection verdict for an individual electronic component."""
    component_name: str
    component_type: str
    status: str  # "OK", "MISSING", "SHIFTED", "ROTATED", "REVIEW"
    confidence: float
    nominal_bbox: Tuple[int, int, int, int]    # (xmin, ymin, xmax, ymax)
    detected_bbox: Tuple[int, int, int, int]
    shift_dx: float
    shift_dy: float
    shift_distance: float
    rotation_deviation_deg: float
    match_score: float
    edge_ratio: float
    anomaly_details: str


class ComponentAnomalyAnalyzer:
    """Deterministic classical CV inspector for SMT assembly placement."""

    def __init__(
        self,
        missing_correlation_threshold: float = 0.45,
        missing_edge_density_ratio: float = 0.40,
        max_allowed_shift_px: float = 14.0,
        rotation_tolerance_deg: float = 12.0,
        search_margin_px: int = 24
    ):
        self.missing_correlation_threshold = missing_correlation_threshold
        self.missing_edge_density_ratio = missing_edge_density_ratio
        self.max_allowed_shift_px = max_allowed_shift_px
        self.rotation_tolerance_deg = rotation_tolerance_deg
        self.search_margin_px = search_margin_px

    def analyze_component(
        self,
        component: ComponentROI,
        template_gray: np.ndarray,
        aligned_test_gray: np.ndarray
    ) -> ComponentInspectionResult:
        """Analyze a single component ROI against nominal reference template."""
        x1, y1, x2, y2 = component.bbox
        h_img, w_img = template_gray.shape[:2]

        # Ensure reference coordinates are bounded
        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(x1 + 1, min(w_img, x2))
        y2 = max(y1 + 1, min(h_img, y2))
        w = x2 - x1
        h = y2 - y1

        template_patch = template_gray[y1:y2, x1:x2]
        if template_patch.size == 0 or template_patch.shape[0] < 4 or template_patch.shape[1] < 4:
            return self._build_review_result(component, "Template patch degenerate.")

        # Nominal test patch at expected location
        test_nominal_patch = aligned_test_gray[y1:y2, x1:x2]

        # Define search neighborhood in aligned test image
        sx1 = max(0, x1 - self.search_margin_px)
        sy1 = max(0, y1 - self.search_margin_px)
        sx2 = min(w_img, x2 + self.search_margin_px)
        sy2 = min(h_img, y2 + self.search_margin_px)
        search_area = aligned_test_gray[sy1:sy2, sx1:sx2]

        if search_area.shape[0] < h or search_area.shape[1] < w:
            return self._build_review_result(component, "Search area smaller than template patch.")

        # 1. Normalized Cross-Correlation Template Matching
        match_map = cv2.matchTemplate(search_area, template_patch, cv2.TM_CCOEFF_NORMED)
        min_val, max_corr, min_loc, max_loc = cv2.minMaxLoc(match_map)

        # Matched coordinates in full-image space
        det_x1 = sx1 + max_loc[0]
        det_y1 = sy1 + max_loc[1]
        det_x2 = det_x1 + w
        det_y2 = det_y1 + h

        dx = float(det_x1 - x1)
        dy = float(det_y1 - y1)
        shift_dist = float(np.sqrt(dx * dx + dy * dy))

        # 2. Structural Edge Density Ratio
        edges_tmpl = cv2.Canny(template_patch, 50, 150)
        edges_test = cv2.Canny(test_nominal_patch, 50, 150)
        cnt_tmpl = float(np.count_nonzero(edges_tmpl))
        cnt_test = float(np.count_nonzero(edges_test))
        edge_ratio = cnt_test / (cnt_tmpl + 1e-5)

        # 3. Orientation / Rotation Analysis via Central Moments
        theta_dev = self._compute_orientation_deviation(template_patch, test_nominal_patch)

        # --- Decision Rules ---

        # Rule 1: Missing Component
        # Correlation collapses significantly or both correlation and edge density drop
        if (max_corr < self.missing_correlation_threshold) or (max_corr < 0.58 and edge_ratio < self.missing_edge_density_ratio):
            conf = min(0.99, max(0.70, 1.0 - max_corr))
            return ComponentInspectionResult(
                component_name=component.name,
                component_type=component.component_type,
                status="MISSING",
                confidence=conf,
                nominal_bbox=(x1, y1, x2, y2),
                detected_bbox=(x1, y1, x2, y2),
                shift_dx=dx,
                shift_dy=dy,
                shift_distance=shift_dist,
                rotation_deviation_deg=theta_dev,
                match_score=float(max_corr),
                edge_ratio=edge_ratio,
                anomaly_details=f"Missing component detected (NCC score {max_corr:.3f} < {self.missing_correlation_threshold}, edge ratio {edge_ratio:.2f})."
            )

        # Rule 2: Rotated Component
        if theta_dev > self.rotation_tolerance_deg:
            conf = min(0.95, 0.65 + 0.30 * min(theta_dev / 45.0, 1.0))
            return ComponentInspectionResult(
                component_name=component.name,
                component_type=component.component_type,
                status="ROTATED",
                confidence=conf,
                nominal_bbox=(x1, y1, x2, y2),
                detected_bbox=(det_x1, det_y1, det_x2, det_y2),
                shift_dx=dx,
                shift_dy=dy,
                shift_distance=shift_dist,
                rotation_deviation_deg=theta_dev,
                match_score=float(max_corr),
                edge_ratio=edge_ratio,
                anomaly_details=f"Rotated component detected (angular deviation {theta_dev:.1f} deg > {self.rotation_tolerance_deg} deg)."
            )

        # Rule 3: Shifted / Misaligned Component
        if shift_dist > self.max_allowed_shift_px:
            conf = min(0.96, 0.60 + 0.35 * min(shift_dist / (self.max_allowed_shift_px * 2), 1.0))
            return ComponentInspectionResult(
                component_name=component.name,
                component_type=component.component_type,
                status="SHIFTED",
                confidence=conf,
                nominal_bbox=(x1, y1, x2, y2),
                detected_bbox=(det_x1, det_y1, det_x2, det_y2),
                shift_dx=dx,
                shift_dy=dy,
                shift_distance=shift_dist,
                rotation_deviation_deg=theta_dev,
                match_score=float(max_corr),
                edge_ratio=edge_ratio,
                anomaly_details=f"Shifted component detected (displacement {shift_dist:.1f}px > {self.max_allowed_shift_px}px; dx={dx:.1f}, dy={dy:.1f})."
            )

        # Rule 4: Normal / PASS
        return ComponentInspectionResult(
            component_name=component.name,
            component_type=component.component_type,
            status="OK",
            confidence=float(max_corr),
            nominal_bbox=(x1, y1, x2, y2),
            detected_bbox=(det_x1, det_y1, det_x2, det_y2),
            shift_dx=dx,
            shift_dy=dy,
            shift_distance=shift_dist,
            rotation_deviation_deg=theta_dev,
            match_score=float(max_corr),
            edge_ratio=edge_ratio,
            anomaly_details="Component verified within nominal placement tolerances."
        )

    def _compute_orientation_deviation(self, tmpl: np.ndarray, test: np.ndarray) -> float:
        """Calculate principal axis orientation deviation using central second-order moments."""
        m_tmpl = cv2.moments(tmpl)
        m_test = cv2.moments(test)

        def angle(m: Dict[str, float]) -> float:
            denom = m["mu20"] - m["mu02"]
            if abs(denom) < 1e-4:
                return 0.0
            return float(0.5 * np.arctan2(2.0 * m["mu11"], denom) * 180.0 / np.pi)

        a_tmpl = angle(m_tmpl)
        a_test = angle(m_test)
        diff = abs(a_tmpl - a_test)
        if diff > 90.0:
            diff = 180.0 - diff
        return float(diff)

    def _build_review_result(self, component: ComponentROI, reason: str) -> ComponentInspectionResult:
        """Generate review container when component ROI cannot be reliably processed."""
        return ComponentInspectionResult(
            component_name=component.name,
            component_type=component.component_type,
            status="REVIEW",
            confidence=0.50,
            nominal_bbox=component.bbox,
            detected_bbox=component.bbox,
            shift_dx=0.0,
            shift_dy=0.0,
            shift_distance=0.0,
            rotation_deviation_deg=0.0,
            match_score=0.0,
            edge_ratio=0.0,
            anomaly_details=f"Inspection review required: {reason}"
        )
