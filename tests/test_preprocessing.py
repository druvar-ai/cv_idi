"""Unit tests for Classical CV Preprocessing Operations."""

import numpy as np
import pytest

from app.preprocessing.filters import apply_gaussian_filter, apply_median_filter, apply_bilateral_filter
from app.preprocessing.enhancement import apply_clahe, apply_contrast_stretching
from app.preprocessing.thresholding import apply_otsu_threshold, apply_fixed_threshold
from app.preprocessing.morphology import morphological_open, morphological_close
from app.preprocessing.pipeline import PreprocessingPipeline


def test_filters():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[25, 25] = 255  # Impulse noise

    gauss = apply_gaussian_filter(img, ksize=(5, 5))
    assert gauss.shape == (50, 50)
    assert gauss[25, 25] < 255  # Energy diffused

    med = apply_median_filter(img, ksize=3)
    assert med[25, 25] == 0  # Impulse noise removed by median filter


def test_enhancement_clahe():
    img = np.random.randint(50, 150, (60, 60), dtype=np.uint8)
    clahe_img = apply_clahe(img, clip_limit=2.0)
    assert clahe_img.shape == (60, 60)
    assert np.std(clahe_img) >= np.std(img) - 5.0


def test_thresholding_otsu():
    # Bimodal distribution with two distinct peaks (around 40 and 190)
    np.random.seed(42)
    p1 = np.random.normal(40, 8, (50, 50)).clip(0, 255).astype(np.uint8)
    p2 = np.random.normal(190, 8, (50, 50)).clip(0, 255).astype(np.uint8)
    bimodal = np.vstack([p1, p2])
    thresh, binary = apply_otsu_threshold(bimodal)
    assert 60 <= thresh <= 150
    assert binary[:50, :].mean() < 50
    assert binary[50:, :].mean() > 200


def test_morphology_open_close():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    mask[5, 5] = 255  # Isolated noise pixel

    opened = morphological_open(mask, kernel_size=3)
    assert opened[5, 5] == 0  # Isolated pixel eliminated
    assert opened[30, 30] == 255  # Core preserved


def test_preprocessing_pipeline():
    pipe = PreprocessingPipeline()
    sample = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    res = pipe.process_bare_pcb(sample)
    assert res.final_image.shape == (640, 640)
    assert "3_denoised" in res.stages
    assert "5_binary" in res.stages
