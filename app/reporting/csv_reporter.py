"""Tabular CSV Report Generator for Industrial Quality Tracking."""

import csv
from pathlib import Path
from typing import List, Union

from app.inspection.decision_engine import StandardizedInspectionVerdict


class CSVReporter:
    """Exports inspection summary and defect items to CSV format."""

    @classmethod
    def export_summary_csv(
        cls,
        verdicts: List[StandardizedInspectionVerdict],
        output_path: Union[str, Path]
    ) -> Path:
        """Export high-level summary rows across multiple inspections."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "inspection_id",
            "timestamp",
            "inspection_type",
            "decision",
            "decision_reason",
            "total_defects",
            "processing_time_ms",
            "quality_inliers",
            "alignment_error_px"
        ]

        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for v in verdicts:
                writer.writerow({
                    "inspection_id": v.inspection_id,
                    "timestamp": v.timestamp,
                    "inspection_type": v.inspection_type,
                    "decision": v.decision,
                    "decision_reason": v.decision_reason,
                    "total_defects": v.total_defect_count,
                    "processing_time_ms": round(v.processing_time_ms, 2),
                    "quality_inliers": v.quality_metrics.get("registration_inliers", 1.0),
                    "alignment_error_px": v.quality_metrics.get("alignment_error_px", 0.0)
                })
        return p

    @classmethod
    def export_defects_csv(
        cls,
        verdict: StandardizedInspectionVerdict,
        output_path: Union[str, Path]
    ) -> Path:
        """Export granular itemized defect records for a single inspection."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "inspection_id",
            "defect_id",
            "category",
            "severity",
            "x1", "y1", "x2", "y2",
            "centroid_x", "centroid_y",
            "area_px",
            "confidence",
            "rule_applied",
            "details"
        ]

        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in verdict.defects:
                writer.writerow({
                    "inspection_id": verdict.inspection_id,
                    "defect_id": d.id,
                    "category": d.category,
                    "severity": d.severity,
                    "x1": d.bounding_box[0],
                    "y1": d.bounding_box[1],
                    "x2": d.bounding_box[2],
                    "y2": d.bounding_box[3],
                    "centroid_x": round(d.centroid[0], 2),
                    "centroid_y": round(d.centroid[1], 2),
                    "area_px": round(d.area, 1),
                    "confidence": round(d.confidence, 3),
                    "rule_applied": d.rule_applied,
                    "details": d.details
                })
        return p
