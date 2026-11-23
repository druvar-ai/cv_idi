"""Morphological Operations for Noise Removal and Structural Clean-up."""

from typing import Tuple
import cv2
import numpy as np


def get_structuring_element(shape: str = "rect", ksize: int = 3) -> np.ndarray:
    """Construct OpenCV structuring element."""
    shapes = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    cv_shape = shapes.get(shape.lower(), cv2.MORPH_RECT)
    k = max(1, ksize if ksize % 2 != 0 else ksize + 1)
    return cv2.getStructuringElement(cv_shape, (k, k))


def morphological_open(
    binary_mask: np.ndarray,
    kernel_size: int = 3,
    shape: str = "rect",
    iterations: int = 1
) -> np.ndarray:
    """Erosion followed by dilation. Removes isolated false-positive noise pixels."""
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morphological_close(
    binary_mask: np.ndarray,
    kernel_size: int = 3,
    shape: str = "rect",
    iterations: int = 1
) -> np.ndarray:
    """Dilation followed by erosion. Closes small pinholes and bridges gaps."""
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def morphological_gradient(
    image: np.ndarray,
    kernel_size: int = 3,
    shape: str = "rect"
) -> np.ndarray:
    """Difference between dilation and erosion (reveals object contours/boundaries)."""
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def morphological_top_hat(
    image: np.ndarray,
    kernel_size: int = 5,
    shape: str = "rect"
) -> np.ndarray:
    """Difference between input and opening (isolates elements smaller than kernel and brighter)."""
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def morphological_black_hat(
    image: np.ndarray,
    kernel_size: int = 5,
    shape: str = "rect"
) -> np.ndarray:
    """Difference between closing and input (isolates dark elements smaller than kernel)."""
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
