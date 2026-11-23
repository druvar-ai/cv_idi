"""Unit tests for Evaluation Metrics and IoU Calculations."""

import pytest
from app.evaluation.metrics import compute_iou, compute_detection_metrics


def test_iou_identical_boxes():
    box_a = (10, 10, 50, 50)
    iou = compute_iou(box_a, box_a)
    assert abs(iou - 1.0) < 1e-4


def test_iou_disjoint_boxes():
    box_a = (0, 0, 10, 10)
    box_b = (20, 20, 30, 30)
    assert compute_iou(box_a, box_b) == 0.0


def test_iou_half_overlap():
    box_a = (0, 0, 20, 20)  # Area 400
    box_b = (10, 0, 30, 20) # Area 400
    # Intersection: (10, 0, 20, 20) -> Area 200
    # Union: 400 + 400 - 200 = 600
    # IoU: 200 / 600 = 1/3
    iou = compute_iou(box_a, box_b)
    assert abs(iou - (1.0 / 3.0)) < 1e-4


def test_detection_metrics_perfect_match():
    boxes = [(10, 10, 50, 50), (60, 60, 100, 100)]
    tp, fp, fn, m_iou = compute_detection_metrics(boxes, boxes, iou_threshold=0.5)
    assert tp == 2
    assert fp == 0
    assert fn == 0
    assert abs(m_iou - 1.0) < 1e-4


def test_detection_metrics_fp_and_fn():
    preds = [(10, 10, 50, 50), (100, 100, 120, 120)] # One match, one extra FP
    gts = [(10, 10, 50, 50), (200, 200, 250, 250)]   # One match, one missed FN
    tp, fp, fn, m_iou = compute_detection_metrics(preds, gts, iou_threshold=0.5)
    assert tp == 1
    assert fp == 1
    assert fn == 1
