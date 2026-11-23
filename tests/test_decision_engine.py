"""Unit tests for Unified Industrial Decision Engine."""

import pytest
from app.config import DecisionEngineConfig
from app.inspection.decision_engine import UnifiedDecisionEngine, StandardizedDefectItem
from app.inspection.bare_pcb.pipeline import BarePCBInspectionResult
from app.inspection.bare_pcb.classifier import ClassifiedDefect


@pytest.fixture
def decision_engine():
    return UnifiedDecisionEngine(config=DecisionEngineConfig())


def test_decision_engine_pass(decision_engine):
    raw_pass = BarePCBInspectionResult(
        inspection_id="TEST-PASS",
        timestamp="2026-11-23T00:00:00Z",
        decision="PASS",
        total_defects=0,
        defects=[],
        registration_info={"is_aligned": True, "inlier_ratio": 0.85, "error_px": 1.2}
    )
    verdict = decision_engine.arbitrate_bare_pcb(raw_pass)
    assert verdict.decision == "PASS"
    assert verdict.total_defect_count == 0


def test_decision_engine_fail_critical(decision_engine):
    defect_open = ClassifiedDefect(
        defect_id=1,
        class_id=1,
        class_name="open",
        bbox=(10, 10, 30, 30),
        centroid=(20.0, 20.0),
        area=150.0,
        confidence=0.92,
        rule_applied="Rule-1: Open Circuit",
        polarity="deficit"
    )
    raw_fail = BarePCBInspectionResult(
        inspection_id="TEST-FAIL",
        timestamp="2026-11-23T00:00:00Z",
        decision="FAIL",
        total_defects=1,
        defects=[defect_open],
        registration_info={"is_aligned": True, "inlier_ratio": 0.90, "error_px": 0.8}
    )
    verdict = decision_engine.arbitrate_bare_pcb(raw_fail)
    assert verdict.decision == "FAIL"
    assert verdict.total_defect_count == 1
    assert verdict.defects[0].severity == "CRITICAL"


def test_decision_engine_review_on_bad_alignment(decision_engine):
    raw_unaligned = BarePCBInspectionResult(
        inspection_id="TEST-REVIEW",
        timestamp="2026-11-23T00:00:00Z",
        decision="PASS",
        total_defects=0,
        defects=[],
        registration_info={"is_aligned": False, "inlier_ratio": 0.15, "error_px": 45.0, "message": "Alignment failed"}
    )
    verdict = decision_engine.arbitrate_bare_pcb(raw_unaligned)
    assert verdict.decision == "REVIEW"
    assert "uncertain" in verdict.decision_reason or "Registration" in verdict.decision_reason
