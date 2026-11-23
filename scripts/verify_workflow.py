"""Comprehensive End-to-End Workflow Audit & Verification Script."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.config import get_config
from app.inspection.bare_pcb.pipeline import BarePCBInspectionPipeline
from app.inspection.assembled_pcb.pipeline import PCBAInspectionPipeline
from app.inspection.decision_engine import UnifiedDecisionEngine
from app.reporting.json_reporter import JSONReporter
from app.reporting.csv_reporter import CSVReporter
from app.reporting.pdf_reporter import PDFReporter
from app.storage.history_manager import HistoryManager


def main() -> None:
    print("=== STARTING COMPLETE WORKFLOW AUDIT ===")

    cfg = get_config()
    history = HistoryManager(Path("outputs/history/inspection_history.db"))
    bare_pipe = BarePCBInspectionPipeline(config=cfg.bare_pcb)
    pcba_pipe = PCBAInspectionPipeline(config=cfg.pcba)
    engine = UnifiedDecisionEngine(config=cfg.decision_engine)

    # 1. BARE PCB - Defective (FAIL)
    ref_bare = cv2.imread("datasets/bare_pcb/templates/00041000_temp.jpg")
    test_bare_def = cv2.imread("datasets/bare_pcb/images/00041000_test.jpg")
    res_bare_fail = bare_pipe.inspect(ref_bare, test_bare_def, inspection_id="AUDIT-BARE-FAIL")
    verdict_bare_fail = engine.arbitrate_bare_pcb(res_bare_fail)
    history.add_inspection(verdict_bare_fail)
    print(f"1. Bare PCB Defective: Decision={verdict_bare_fail.decision}, Defects={verdict_bare_fail.total_defect_count}")
    print(f"   Reason: {verdict_bare_fail.decision_reason}")
    assert verdict_bare_fail.decision == "FAIL", f"Expected FAIL, got {verdict_bare_fail.decision}"
    assert verdict_bare_fail.total_defect_count > 0

    # Verify intermediate images saved
    inter_bare_dir = Path("outputs/intermediate/AUDIT-BARE-FAIL")
    assert inter_bare_dir.exists(), "Intermediate directory was not created!"
    bare_saved_files = sorted([f.name for f in inter_bare_dir.glob("*.png")])
    print(f"   Intermediate images saved ({len(bare_saved_files)}): {bare_saved_files}")
    assert len(bare_saved_files) >= 7

    # Generate reports
    p_json = Path("outputs/reports/AUDIT-BARE-FAIL.json")
    p_csv = Path("outputs/reports/AUDIT-BARE-FAIL_defects.csv")
    p_pdf = Path("outputs/reports/AUDIT-BARE-FAIL.pdf")
    JSONReporter.save_report(verdict_bare_fail, p_json)
    CSVReporter.export_defects_csv(verdict_bare_fail, p_csv)
    PDFReporter.generate_pdf_report(verdict_bare_fail, p_pdf)
    assert p_json.exists() and p_json.stat().st_size > 100
    assert p_csv.exists() and p_csv.stat().st_size > 50
    assert p_pdf.exists() and p_pdf.stat().st_size > 1000
    print(f"   Reports verified: JSON ({p_json.stat().st_size} B), CSV ({p_csv.stat().st_size} B), PDF ({p_pdf.stat().st_size} B)")

    # 2. BARE PCB - Reference vs Reference (PASS)
    res_bare_pass = bare_pipe.inspect(ref_bare, ref_bare, inspection_id="AUDIT-BARE-PASS")
    verdict_bare_pass = engine.arbitrate_bare_pcb(res_bare_pass)
    history.add_inspection(verdict_bare_pass)
    print(f"2. Bare PCB Reference (Defect-Free): Decision={verdict_bare_pass.decision}, Defects={verdict_bare_pass.total_defect_count}")
    assert verdict_bare_pass.decision == "PASS"
    assert verdict_bare_pass.total_defect_count == 0

    # 3. PCBA - Defective (FAIL)
    ref_pcba = cv2.imread("datasets/assembled_pcb/templates/template_pcba_01.jpg")
    test_pcba_def = cv2.imread("datasets/assembled_pcb/images/test_pcba_01_defective.jpg")
    spec_xml = Path("datasets/assembled_pcb/annotations/template_pcba_01.xml")
    res_pcba_fail = pcba_pipe.inspect(ref_pcba, test_pcba_def, components_spec=spec_xml, inspection_id="AUDIT-PCBA-FAIL")
    verdict_pcba_fail = engine.arbitrate_pcba(res_pcba_fail)
    history.add_inspection(verdict_pcba_fail)
    print(f"3. PCBA Defective: Decision={verdict_pcba_fail.decision}, Anomalies={verdict_pcba_fail.total_defect_count}")
    print(f"   Reason: {verdict_pcba_fail.decision_reason}")
    assert verdict_pcba_fail.decision == "FAIL"
    assert verdict_pcba_fail.total_defect_count > 0

    # Verify intermediate images saved
    inter_pcba_dir = Path("outputs/intermediate/AUDIT-PCBA-FAIL")
    assert inter_pcba_dir.exists(), "PCBA intermediate directory not created!"
    pcba_saved_files = sorted([f.name for f in inter_pcba_dir.glob("*.png")])
    print(f"   Intermediate images saved ({len(pcba_saved_files)}): {pcba_saved_files}")
    assert len(pcba_saved_files) >= 5

    # Generate reports
    p_json = Path("outputs/reports/AUDIT-PCBA-FAIL.json")
    p_csv = Path("outputs/reports/AUDIT-PCBA-FAIL_defects.csv")
    p_pdf = Path("outputs/reports/AUDIT-PCBA-FAIL.pdf")
    JSONReporter.save_report(verdict_pcba_fail, p_json)
    CSVReporter.export_defects_csv(verdict_pcba_fail, p_csv)
    PDFReporter.generate_pdf_report(verdict_pcba_fail, p_pdf)
    assert p_json.exists() and p_json.stat().st_size > 100
    assert p_csv.exists() and p_csv.stat().st_size > 50
    assert p_pdf.exists() and p_pdf.stat().st_size > 1000
    print(f"   Reports verified: JSON ({p_json.stat().st_size} B), CSV ({p_csv.stat().st_size} B), PDF ({p_pdf.stat().st_size} B)")

    # 4. PCBA - Reference vs Reference (PASS)
    res_pcba_pass = pcba_pipe.inspect(ref_pcba, ref_pcba, components_spec=spec_xml, inspection_id="AUDIT-PCBA-PASS")
    verdict_pcba_pass = engine.arbitrate_pcba(res_pcba_pass)
    history.add_inspection(verdict_pcba_pass)
    print(f"4. PCBA Reference (Defect-Free): Decision={verdict_pcba_pass.decision}, Anomalies={verdict_pcba_pass.total_defect_count}")
    assert verdict_pcba_pass.decision == "PASS"
    assert verdict_pcba_pass.total_defect_count == 0

    # 5. REVIEW Outcome Test (Mismatched alignment)
    # Poorly matched board triggering inlier ratio or alignment error violation
    blank_noisy = np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)
    res_bare_review = bare_pipe.inspect(ref_bare, blank_noisy, inspection_id="AUDIT-BARE-REVIEW")
    verdict_bare_review = engine.arbitrate_bare_pcb(res_bare_review)
    history.add_inspection(verdict_bare_review)
    print(f"5. Mismatched Input Check: Decision={verdict_bare_review.decision}")
    print(f"   Reason: {verdict_bare_review.decision_reason}")
    assert verdict_bare_review.decision == "REVIEW", f"Expected REVIEW, got {verdict_bare_review.decision}"

    # 6. Verify SQLite History
    records = history.list_inspections(limit=10)
    stats = history.get_summary_stats()
    print(f"6. SQLite History Persistence: Retrieved {len(records)} recent records, Total={stats['total_inspections']}, Pass={stats['pass_count']}, Fail={stats['fail_count']}, Review={stats['review_count']}")
    for r in records[:5]:
        print(f"   • ID: {r['inspection_id']} | Type: {r['inspection_type']} | Decision: {r['decision']} | Defects: {r['total_defect_count']}")

    print("\n>>> ALL WORKFLOW TESTS & AUDITS SUCCESSFULLY PASSED <<<")


if __name__ == "__main__":
    main()
