"""Binarization and Thresholding Operators."""

from typing import Tuple
import cv2
import numpy as np


def apply_otsu_threshold(image_gray: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute global optimal threshold using Otsu's variance maximization."""
    if len(image_gray.shape) != 2:
        raise ValueError("Otsu threshold requires a single-channel grayscale image.")
    thresh_val, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thresh_val), binary


def apply_adaptive_threshold(
    image_gray: np.ndarray,
    method: str = "gaussian",
    block_size: int = 11,
    c_constant: float = 2.0
) -> np.ndarray:
    """Apply local adaptive thresholding."""
    if len(image_gray.shape) != 2:
        raise ValueError("Adaptive threshold requires a single-channel grayscale image.")
    bs = block_size if block_size % 2 != 0 else block_size + 1
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method.lower() == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    return cv2.adaptiveThreshold(image_gray, 255, adapt_type, cv2.THRESH_BINARY, bs, c_constant)


def apply_fixed_threshold(
    image_gray: np.ndarray,
    threshold: int = 128
) -> np.ndarray:
    """Apply standard fixed value thresholding."""
    _, binary = cv2.threshold(image_gray, threshold, 255, cv2.THRESH_BINARY)
    return binary
