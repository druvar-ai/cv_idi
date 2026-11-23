"""Unit tests for Image Acquisition, Validation and Loading."""

import numpy as np
import pytest
from pathlib import Path

from app.acquisition.validator import ImageValidator, ValidationResult
from app.acquisition.loader import ImageLoader


def test_validator_file_not_found():
    res = ImageValidator.validate_file_path(Path("non_existent_image.png"))
    assert not res.is_valid
    assert res.status == "FILE_NOT_FOUND"


def test_validator_unsupported_extension(tmp_path):
    bad_file = tmp_path / "test.txt"
    bad_file.write_text("not an image")
    res = ImageValidator.validate_file_path(bad_file)
    assert not res.is_valid
    assert res.status == "UNSUPPORTED_FORMAT"


def test_validator_empty_file(tmp_path):
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")
    res = ImageValidator.validate_file_path(empty_file)
    assert not res.is_valid
    assert res.status == "EMPTY_FILE"


def test_validator_blank_low_contrast():
    solid_black = np.zeros((100, 100), dtype=np.uint8)
    res = ImageValidator.validate_image_array(solid_black)
    assert not res.is_valid
    assert res.status == "LOW_CONTRAST_OR_BLANK"


def test_validator_valid_image():
    # Synthetic random image with high variance
    np.random.seed(42)
    valid_arr = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    res = ImageValidator.validate_image_array(valid_arr)
    assert res.is_valid
    assert res.status == "OK"
    assert res.dimensions == (120, 120, 3)


def test_loader_real_dataset_image():
    p = Path("datasets/bare_pcb/templates/00041000_temp.jpg")
    img, val = ImageLoader.load_from_path(p)
    assert val.is_valid
    assert img is not None
    assert img.dimensions == (640, 640, 3)
    assert img.image_gray.shape == (640, 640)
