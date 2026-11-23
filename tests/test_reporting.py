"""Unit tests for JSON, CSV, and PDF Report Generation."""

from pathlib import Path
import json
import pytest

from app.inspection.decision_engine import StandardizedInspectionVerdict, StandardizedDefectItem
from app.reporting.json_reporter import JSONReporter
from app.reporting.csv_reporter import CSVReporter
from app.reporting.pdf_reporter import PDFReporter


@pytest.fixture
def sample_verdict():
    defect = StandardizedDefectItem(
        id=1,
        category="open",
        severity="CRITICAL",
        bounding_box=(50, 60, 80, 90),
        centroid=(65.0, 75.0),
        area=120.0,
        confidence=0.94,
        rule_applied="Rule-1: Open Trace",
        details="Trace discontinuity detected"
    )
    return StandardizedInspectionVerdict(
        inspection_id="INSP-TEST-001",
        timestamp="2026-11-23T12:00:00Z",
        inspection_type="Bare PCB Fabrication",
        decision="FAIL",
        decision_reason="Detected 1 manufacturing defect (open circuit)",
        total_defect_count=1,
        defect_summary={"open": 1},
        defects=[defect],
        processing_time_ms=85.4,
        quality_metrics={"registration_inliers": 0.95, "alignment_error_px": 0.5},
        algorithm_info={"pipeline": "BarePCBInspectionPipeline"}
    )


def test_json_reporting(sample_verdict, tmp_path):
    out_file = tmp_path / "test.json"
    saved = JSONReporter.save_report(sample_verdict, out_file)
    assert saved.exists()
    with open(saved, "r") as f:
        data = json.load(f)
    assert data["inspection_id"] == "INSP-TEST-001"
    assert data["decision"]["status"] == "FAIL"
    assert len(data["defects_catalog"]) == 1


def test_csv_reporting(sample_verdict, tmp_path):
    out_csv = tmp_path / "test.csv"
    saved = CSVReporter.export_defects_csv(sample_verdict, out_csv)
    assert saved.exists()
    content = saved.read_text()
    assert "INSP-TEST-001" in content
    assert "open" in content
    assert "CRITICAL" in content


def test_pdf_reporting(sample_verdict, tmp_path):
    out_pdf = tmp_path / "test.pdf"
    saved = PDFReporter.generate_pdf_report(sample_verdict, out_pdf)
    assert saved.exists()
    assert saved.stat().st_size > 1000
    # Check valid PDF magic bytes
    with open(saved, "rb") as f:
        header = f.read(4)
    assert header == b"%PDF"
