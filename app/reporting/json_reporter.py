"""Structured JSON Report Generator for MES / ERP Integration."""

import json
from pathlib import Path
from typing import Any, Dict, Union

from app.inspection.decision_engine import StandardizedInspectionVerdict


class JSONReporter:
    """Generates comprehensive structured JSON audit reports."""

    @classmethod
    def generate_report_dict(cls, verdict: StandardizedInspectionVerdict) -> Dict[str, Any]:
        """Convert inspection verdict to standardized dictionary."""
        return {
            "report_schema": "AOI-PCB-Phase1-v1.0",
            "inspection_id": verdict.inspection_id,
            "timestamp": verdict.timestamp,
            "domain": verdict.inspection_type,
            "decision": {
                "status": verdict.decision,
                "reason": verdict.decision_reason,
                "total_defects": verdict.total_defect_count,
                "summary": verdict.defect_summary
            },
            "performance": {
                "processing_time_ms": verdict.processing_time_ms,
                "latency_sec": verdict.processing_time_ms / 1000.0,
                "estimated_throughput_fps": round(1000.0 / max(1.0, verdict.processing_time_ms), 1)
            },
            "quality_assurance": verdict.quality_metrics,
            "algorithm_specification": verdict.algorithm_info,
            "defects_catalog": [
                {
                    "defect_id": d.id,
                    "category": d.category,
                    "severity": d.severity,
                    "bounding_box_ltrb": list(d.bounding_box),
                    "centroid": list(d.centroid),
                    "area_pixels": d.area,
                    "confidence_score": d.confidence,
                    "rule_applied": d.rule_applied,
                    "technical_details": d.details
                }
                for d in verdict.defects
            ]
        }

    @classmethod
    def save_report(
        cls,
        verdict: StandardizedInspectionVerdict,
        output_path: Union[str, Path]
    ) -> Path:
        """Write report to JSON file on disk."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        report_data = cls.generate_report_dict(verdict)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        return p
