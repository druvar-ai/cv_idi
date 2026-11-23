"""Bare PCB Evaluation Runner against DeepPCB Ground Truth."""

import glob
import os
from pathlib import Path
import time
from typing import Dict, List, Optional
import cv2
import numpy as np

from app.config import get_config
from app.inspection.bare_pcb.pipeline import BarePCBInspectionPipeline
from .metrics import compute_iou, compute_detection_metrics, EvaluationMetrics


class BarePCBEvaluator:
    """Evaluates classical CV pipeline on DeepPCB test split."""

    DEEPPCB_CLASS_NAMES = {
        1: "open",
        2: "short",
        3: "mousebite",
        4: "spur",
        5: "copper",
        6: "pin-hole"
    }

    def __init__(self, dataset_dir: Path = Path("datasets/bare_pcb")):
        self.dataset_dir = Path(dataset_dir)
        cfg = get_config()
        self.pipeline = BarePCBInspectionPipeline(config=cfg.bare_pcb)

    def evaluate(
        self,
        max_samples: Optional[int] = None,
        iou_threshold: float = 0.33
    ) -> EvaluationMetrics:
        """Run benchmark evaluation over curated DeepPCB dataset pairs."""
        anno_files = sorted(glob.glob(str(self.dataset_dir / "annotations" / "*.txt")))
        if max_samples:
            anno_files = anno_files[:max_samples]

        total_tp = 0
        total_fp = 0
        total_fn = 0
        all_matched_ious: List[float] = []
        latencies_ms: List[float] = []

        class_stats: Dict[str, Dict[str, int]] = {
            name: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
            for name in self.DEEPPCB_CLASS_NAMES.values()
        }

        for txt_path in anno_files:
            base_id = Path(txt_path).stem
            test_img_path = self.dataset_dir / "images" / f"{base_id}_test.jpg"
            temp_img_path = self.dataset_dir / "templates" / f"{base_id}_temp.jpg"

            if not test_img_path.exists() or not temp_img_path.exists():
                continue

            # Load Ground Truth
            gt_boxes: List[Tuple[int, int, int, int]] = []
            gt_classes: List[str] = []
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        x1, y1, x2, y2 = [int(v) for v in parts[:4]]
                        type_id = int(parts[4])
                        c_name = self.DEEPPCB_CLASS_NAMES.get(type_id, "unknown")
                        gt_boxes.append((x1, y1, x2, y2))
                        gt_classes.append(c_name)
                        if c_name in class_stats:
                            class_stats[c_name]["gt"] += 1

            # Run Pipeline
            test_bgr = cv2.imread(str(test_img_path))
            temp_bgr = cv2.imread(str(temp_img_path))

            t0 = time.perf_counter()
            res = self.pipeline.inspect(reference_bgr=temp_bgr, test_bgr=test_bgr, inspection_id=base_id)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            pred_boxes = [d.bbox for d in res.defects]
            pred_classes = [d.class_name for d in res.defects]

            for c in pred_classes:
                if c in class_stats:
                    class_stats[c]["pred"] += 1

            tp, fp, fn, m_iou = compute_detection_metrics(
                pred_boxes=pred_boxes,
                gt_boxes=gt_boxes,
                iou_threshold=iou_threshold
            )

            # Per-class matching
            matched_gt_set = set()
            for p_idx, p_box in enumerate(pred_boxes):
                p_cls = pred_classes[p_idx]
                best_iou = 0.0
                best_g_idx = -1
                for g_idx, g_box in enumerate(gt_boxes):
                    if g_idx in matched_gt_set:
                        continue
                    iou = compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_g_idx = g_idx
                if best_iou >= iou_threshold and best_g_idx >= 0:
                    matched_gt_set.add(best_g_idx)
                    g_cls = gt_classes[best_g_idx]
                    if p_cls == g_cls and p_cls in class_stats:
                        class_stats[p_cls]["tp"] += 1

            total_tp += tp
            total_fp += fp
            total_fn += fn
            if m_iou > 0:
                all_matched_ious.append(m_iou)

        # Compute aggregate metrics
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = (2.0 * precision * recall) / (precision + recall + 1e-6)
        det_rate = recall
        fpr = total_fp / (total_fp + total_tp + 1e-6)
        mean_iou_val = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0
        mean_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0

        # Compute per-class precision / recall
        per_class: Dict[str, Dict[str, float]] = {}
        for c_name, s in class_stats.items():
            c_p = s["tp"] / (s["pred"] + 1e-6) if s["pred"] > 0 else 0.0
            c_r = s["tp"] / (s["gt"] + 1e-6) if s["gt"] > 0 else 0.0
            per_class[c_name] = {
                "ground_truth": s["gt"],
                "predicted": s["pred"],
                "precision": round(c_p, 3),
                "recall": round(c_r, 3)
            }

        return EvaluationMetrics(
            true_positives=total_tp,
            false_positives=total_fp,
            true_negatives=len(anno_files),  # Defect-free areas/boards
            false_negatives=total_fn,
            total_ground_truth=total_tp + total_fn,
            total_predictions=total_tp + total_fp,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            detection_rate=round(det_rate, 4),
            false_positive_rate=round(fpr, 4),
            mean_iou=round(mean_iou_val, 4),
            mean_latency_ms=round(mean_lat, 2),
            per_class_metrics=per_class
        )
