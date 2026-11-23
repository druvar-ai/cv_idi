"""Inspection History Manager and Persistent Audit Database."""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import os
import sqlite3
from typing import Any, Dict, List, Optional
import numpy as np

from app.inspection.decision_engine import StandardizedInspectionVerdict, StandardizedDefectItem


class HistoryManager:
    """Manages SQLite and JSON inspection history audit trail."""

    def __init__(self, db_path: Path = Path("outputs/history/inspection_history.db")):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inspections (
                    inspection_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    inspection_type TEXT,
                    decision TEXT,
                    decision_reason TEXT,
                    total_defect_count INTEGER,
                    defect_summary_json TEXT,
                    defects_json TEXT,
                    processing_time_ms REAL,
                    quality_metrics_json TEXT,
                    algorithm_info_json TEXT
                )
            """)
            conn.commit()

    def add_inspection(self, verdict: StandardizedInspectionVerdict) -> None:
        """Persist a new inspection outcome to database."""
        # Convert defect objects to serializable dicts
        defects_list = [
            {
                "id": d.id,
                "category": d.category,
                "severity": d.severity,
                "bounding_box": list(d.bounding_box),
                "centroid": list(d.centroid),
                "area": float(d.area),
                "confidence": float(d.confidence),
                "rule_applied": d.rule_applied,
                "details": d.details,
            }
            for d in verdict.defects
        ]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO inspections (
                    inspection_id, timestamp, inspection_type, decision, decision_reason,
                    total_defect_count, defect_summary_json, defects_json,
                    processing_time_ms, quality_metrics_json, algorithm_info_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                verdict.inspection_id,
                verdict.timestamp,
                verdict.inspection_type,
                verdict.decision,
                verdict.decision_reason,
                verdict.total_defect_count,
                json.dumps(verdict.defect_summary),
                json.dumps(defects_list),
                verdict.processing_time_ms,
                json.dumps(verdict.quality_metrics),
                json.dumps(verdict.algorithm_info),
            ))
            conn.commit()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Aggregate high-level production metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM inspections")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM inspections WHERE inspection_type LIKE '%Bare%'")
            bare_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM inspections WHERE inspection_type LIKE '%Assembled%'")
            pcba_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM inspections WHERE decision = 'PASS'")
            pass_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM inspections WHERE decision = 'FAIL'")
            fail_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM inspections WHERE decision = 'REVIEW'")
            review_count = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(processing_time_ms) FROM inspections")
            avg_time = cursor.fetchone()[0] or 0.0

        return {
            "total_inspections": total,
            "bare_pcb_inspections": bare_count,
            "assembled_pcb_inspections": pcba_count,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "review_count": review_count,
            "pass_rate_percent": (pass_count / total * 100.0) if total > 0 else 0.0,
            "avg_latency_ms": float(avg_time),
        }

    def list_inspections(
        self,
        limit: int = 50,
        inspection_type: Optional[str] = None,
        decision: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve historical inspections with optional filters."""
        query = "SELECT inspection_id, timestamp, inspection_type, decision, decision_reason, total_defect_count, defect_summary_json, defects_json, processing_time_ms, quality_metrics_json, algorithm_info_json FROM inspections WHERE 1=1"
        params = []
        if inspection_type:
            query += " AND inspection_type = ?"
            params.append(inspection_type)
        if decision:
            query += " AND decision = ?"
            params.append(decision)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            for row in cursor.fetchall():
                results.append({
                    "inspection_id": row[0],
                    "timestamp": row[1],
                    "inspection_type": row[2],
                    "decision": row[3],
                    "decision_reason": row[4],
                    "total_defect_count": row[5],
                    "defect_summary": json.loads(row[6]) if row[6] else {},
                    "defects": json.loads(row[7]) if row[7] else [],
                    "processing_time_ms": row[8],
                    "quality_metrics": json.loads(row[9]) if row[9] else {},
                    "algorithm_info": json.loads(row[10]) if row[10] else {},
                })
        return results
