"""Unit and Integration tests for Bare PCB Inspection Pipeline."""

import cv2
import numpy as np
import pytest
from pathlib import Path

from app.config import get_config
from app.inspection.bare_pcb.pipeline import BarePCBInspectionPipeline
from app.inspection.bare_pcb.diff_analyzer import DifferenceAnalyzer


@pytest.fixture
def bare_pipeline():
    cfg = get_config()
    return BarePCBInspectionPipeline(config=cfg.bare_pcb)


def test_bare_pcb_pass_on_identical(bare_pipeline):
    ref_path = Path("datasets/bare_pcb/templates/00041000_temp.jpg")
    ref_bgr = cv2.imread(str(ref_path))
    assert ref_bgr is not None

    res = bare_pipeline.inspect(reference_bgr=ref_bgr, test_bgr=ref_bgr.copy())
    assert res.decision == "PASS"
    assert res.total_defects == 0
    assert res.binary_defect_mask is not None
    assert cv2.countNonZero(res.binary_defect_mask) == 0


def test_bare_pcb_fail_on_defective(bare_pipeline):
    ref_path = Path("datasets/bare_pcb/templates/00041000_temp.jpg")
    test_path = Path("datasets/bare_pcb/images/00041000_test.jpg")
    ref_bgr = cv2.imread(str(ref_path))
    test_bgr = cv2.imread(str(test_path))

    res = bare_pipeline.inspect(reference_bgr=ref_bgr, test_bgr=test_bgr)
    assert res.decision == "FAIL"
    assert res.total_defects > 0
    assert len(res.defects) == res.total_defects
    assert "open" in res.defect_counts_by_type
    assert "pin-hole" in res.defect_counts_by_type


def test_difference_analyzer():
    analyzer = DifferenceAnalyzer(morph_kernel_size=3, morph_iterations=1)
    tmpl = np.zeros((100, 100), dtype=np.uint8)
    test = np.zeros((100, 100), dtype=np.uint8)

    tmpl[30:70, 30:70] = 255  # Copper trace in template
    test[30:70, 30:70] = 255  # Copper trace in test
    test[45:55, 45:55] = 0    # Missing copper (deficit / pinhole / open)
    test[10:20, 10:20] = 255  # Extra copper (excess / spurious copper)

    diff = analyzer.analyze(tmpl, test)
    assert cv2.countNonZero(diff.copper_deficit_clean) > 0
    assert cv2.countNonZero(diff.copper_excess_clean) > 0
    assert diff.diff_pixel_count > 0
