"""Unit and Integration tests for Assembled PCB (PCBA) Inspection."""

import cv2
import numpy as np
import pytest
from pathlib import Path

from app.config import get_config
from app.inspection.assembled_pcb.pipeline import PCBAInspectionPipeline
from app.inspection.assembled_pcb.component_detector import ComponentSpecificationLoader


@pytest.fixture
def pcba_pipeline():
    cfg = get_config()
    return PCBAInspectionPipeline(config=cfg.pcba)


def test_pcba_pass_on_identical(pcba_pipeline):
    ref_path = Path("datasets/assembled_pcb/templates/template_pcba_01.jpg")
    xml_path = Path("datasets/assembled_pcb/annotations/template_pcba_01.xml")
    ref_bgr = cv2.imread(str(ref_path))

    res = pcba_pipeline.inspect(reference_bgr=ref_bgr, test_bgr=ref_bgr.copy(), components_spec=xml_path)
    assert res.decision == "PASS"
    assert res.total_anomalies == 0
    assert res.anomaly_counts_by_type["OK"] == 60


def test_pcba_fail_on_defective(pcba_pipeline):
    ref_path = Path("datasets/assembled_pcb/templates/template_pcba_01.jpg")
    test_path = Path("datasets/assembled_pcb/images/test_pcba_01_defective.jpg")
    xml_path = Path("datasets/assembled_pcb/annotations/template_pcba_01.xml")

    ref_bgr = cv2.imread(str(ref_path))
    test_bgr = cv2.imread(str(test_path))

    res = pcba_pipeline.inspect(reference_bgr=ref_bgr, test_bgr=test_bgr, components_spec=xml_path)
    assert res.decision == "FAIL"
    assert res.total_anomalies > 0

    # Specifically verify missing component 53_ac was caught
    comp_53 = next((c for c in res.component_results if c.component_name == "53_ac"), None)
    assert comp_53 is not None
    assert comp_53.status == "MISSING"


def test_xml_spec_loader():
    xml_path = Path("datasets/assembled_pcb/annotations/template_pcba_01.xml")
    comps = ComponentSpecificationLoader.load_from_xml(xml_path)
    assert len(comps) == 60
    assert comps[0].name == "1_a"
    assert comps[0].width > 0
    assert comps[0].height > 0
