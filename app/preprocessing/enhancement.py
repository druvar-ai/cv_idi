"""Contrast Enhancement and Photometric Normalization."""

from typing import Tuple
import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 2:
        return clahe.apply(image)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Equalize only the luminance channel in LAB space to preserve color balance
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    else:
        return image.copy()


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply global histogram equalization."""
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Equalize in YCrCb luminance
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        return cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2BGR)
    return image.copy()


def apply_contrast_stretching(
    image: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0
) -> np.ndarray:
    """Min-max percentile contrast stretching."""
    in_min = np.percentile(image, lower_percentile)
    in_max = np.percentile(image, upper_percentile)
    if in_max <= in_min:
        return image.copy()
    stretched = (image.astype(np.float32) - in_min) * (255.0 / (in_max - in_min))
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return stretched
