"""Computer Vision Performance Metrics and IoU Matching."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = float(inter_w * inter_h)

    area_a = float(max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1]))
    area_b = float(max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1]))

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


@dataclass
class EvaluationMetrics:
    """Rigorous confusion matrix and detection metrics."""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    total_ground_truth: int = 0
    total_predictions: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    mean_iou: float = 0.0
    mean_latency_ms: float = 0.0
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


def compute_detection_metrics(
    pred_boxes: List[Tuple[int, int, int, int]],
    gt_boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.33,
    pred_classes: Optional[List[str]] = None,
    gt_classes: Optional[List[str]] = None
) -> Tuple[int, int, int, float]:
    """Greedy bipartite matching between predicted and ground truth boxes.
    Returns: (tp, fp, fn, mean_iou_matched).
    """
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0, 1.0
    if not gt_boxes and pred_boxes:
        return 0, len(pred_boxes), 0, 0.0
    if gt_boxes and not pred_boxes:
        return 0, 0, len(gt_boxes), 0.0

    matched_gt = set()
    matched_pred = set()
    ious_matched = []

    # Sort predicted boxes (by class match if provided)
    for p_idx, p_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            # Check class match if available
            if pred_classes and gt_classes:
                if pred_classes[p_idx] == gt_classes[best_gt_idx]:
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(p_idx)
                    ious_matched.append(best_iou)
            else:
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
                ious_matched.append(best_iou)

    tp = len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)
    mean_iou = float(np.mean(ious_matched)) if ious_matched else 0.0

    return tp, fp, fn, mean_iou
