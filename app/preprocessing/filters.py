"""Image Filtering and Noise Reduction Operators."""

from typing import Tuple
import cv2
import numpy as np


def apply_gaussian_filter(
    image: np.ndarray,
    ksize: Tuple[int, int] = (5, 5),
    sigma: float = 1.0
) -> np.ndarray:
    """Apply Gaussian spatial filtering to reduce high-frequency Gaussian noise."""
    # Ensure kernel dimensions are positive odd integers
    kw = ksize[0] if ksize[0] % 2 != 0 else ksize[0] + 1
    kh = ksize[1] if ksize[1] % 2 != 0 else ksize[1] + 1
    return cv2.GaussianBlur(image, (kw, kh), sigmaX=sigma, sigmaY=sigma)


def apply_median_filter(
    image: np.ndarray,
    ksize: int = 3
) -> np.ndarray:
    """Apply non-linear median filtering for salt-and-pepper / impulse noise."""
    k = ksize if ksize % 2 != 0 else ksize + 1
    return cv2.medianBlur(image, k)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """Apply edge-preserving bilateral filtering."""
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def filter_image(
    image: np.ndarray,
    method: str = "gaussian",
    **kwargs
) -> np.ndarray:
    """Unified filtering dispatcher."""
    m = method.lower()
    if m == "gaussian":
        return apply_gaussian_filter(image, **kwargs)
    elif m == "median":
        return apply_median_filter(image, **kwargs)
    elif m == "bilateral":
        return apply_bilateral_filter(image, **kwargs)
    elif m == "none":
        return image.copy()
    else:
        raise ValueError(f"Unknown filtering method '{method}'. Choose 'gaussian', 'median', 'bilateral', or 'none'.")
