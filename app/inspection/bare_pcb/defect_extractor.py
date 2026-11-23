"""Defect Candidate Extraction and Geometric Feature Measurement."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class DefectCandidate:
    """Quantitative geometric feature vector for a detected defect candidate blob."""
    id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: Tuple[float, float]    # (cx, cy)
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    solidity: float
    polarity: str                    # "deficit" or "excess"
    contour: np.ndarray
    surrounding_copper_ratio: float
    touches_copper_boundary: bool


class DefectExtractor:
    """Extracts connected component regions and measures structural features."""

    def __init__(
        self,
        min_area: int = 25,
        max_area: int = 15000,
        bbox_padding: int = 8
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.bbox_padding = bbox_padding

    def extract_candidates(
        self,
        difference_mask: np.ndarray,
        polarity: str,
        template_binary: np.ndarray
    ) -> List[DefectCandidate]:
        """Extract defect blobs from binary difference map with topological features."""
        candidates: List[DefectCandidate] = []
        if difference_mask is None or cv2.countNonZero(difference_mask) == 0:
            return candidates

        h_img, w_img = difference_mask.shape[:2]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            difference_mask, connectivity=8
        )

        # In DeepPCB: Copper is dark (< 128)
        template_copper = (template_binary < 128).astype(np.uint8) * 255

        for label_idx in range(1, num_labels):
            area = float(stats[label_idx, cv2.CC_STAT_AREA])
            if area < self.min_area or area > self.max_area:
                continue

            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            cx, cy = float(centroids[label_idx][0]), float(centroids[label_idx][1])

            # Apply bounded padding to bbox to match ground truth envelope
            x1 = max(0, x - self.bbox_padding)
            y1 = max(0, y - self.bbox_padding)
            x2 = min(w_img, x + w + self.bbox_padding)
            y2 = min(h_img, y + h + self.bbox_padding)

            # Extract single-component contour
            blob_mask = (labels == label_idx).astype(np.uint8) * 255
            contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = contours[0]

            perimeter = float(cv2.arcLength(cnt, closed=True))
            circularity = (4.0 * np.pi * area) / ((perimeter * perimeter) + 1e-5)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / (hull_area + 1e-5)

            # Analyze surrounding template copper neighborhood
            roi_y1 = max(0, y1 - 4)
            roi_y2 = min(h_img, y2 + 4)
            roi_x1 = max(0, x1 - 4)
            roi_x2 = min(w_img, x2 + 4)

            template_roi = template_copper[roi_y1:roi_y2, roi_x1:roi_x2]
            copper_pixels = np.count_nonzero(template_roi > 127)
            total_pixels = template_roi.size
            copper_ratio = float(copper_pixels) / float(total_pixels + 1e-5)

            # Check if blob touches copper trace edge
            dilated_blob = cv2.dilate(blob_mask, np.ones((5, 5), np.uint8))
            touches_edge = bool(np.any((dilated_blob > 0) & (template_copper > 127)))

            cand = DefectCandidate(
                id=label_idx,
                bbox=(x1, y1, x2, y2),
                centroid=(cx, cy),
                area=area,
                perimeter=perimeter,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                solidity=solidity,
                polarity=polarity,
                contour=cnt,
                surrounding_copper_ratio=copper_ratio,
                touches_copper_boundary=touches_edge
            )
            candidates.append(cand)

        return candidates
