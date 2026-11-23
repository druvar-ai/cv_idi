"""Dual-Polarity Image Difference Analyzer for Bare PCB Inspection."""

from dataclasses import dataclass
from typing import Dict, Tuple
import cv2
import numpy as np


@dataclass
class DifferenceResult:
    """Encapsulates raw and filtered dual-polarity difference maps."""
    copper_deficit_raw: np.ndarray    # Missing copper (open, mousebite, pin-hole)
    copper_excess_raw: np.ndarray     # Extra copper (short, spur, spurious copper)
    combined_raw: np.ndarray
    copper_deficit_clean: np.ndarray  # Morphologically filtered
    copper_excess_clean: np.ndarray
    combined_clean: np.ndarray
    diff_pixel_count: int


class DifferenceAnalyzer:
    """Computes directional subtractive differences between template and test PCB."""

    def __init__(
        self,
        morph_kernel_size: int = 3,
        morph_close_kernel_size: int = 7,
        morph_iterations: int = 1
    ):
        self.kernel_open = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (morph_kernel_size, morph_kernel_size)
        )
        self.kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (morph_close_kernel_size, morph_close_kernel_size)
        )
        self.morph_iterations = morph_iterations

    def analyze(
        self,
        template_binary: np.ndarray,
        aligned_test_binary: np.ndarray
    ) -> DifferenceResult:
        """Compute dual-polarity difference accounting for linear scan CCD photometry.
        In DeepPCB: Conductive copper traces are dark (< 128), substrate is bright (>= 128).
        """
        # Convert to strict binary copper presence masks (Copper = 255, Substrate = 0)
        t_copper = (template_binary < 128).astype(np.uint8) * 255
        a_copper = (aligned_test_binary < 128).astype(np.uint8) * 255

        # Copper Deficit (Missing Copper): Open circuit, Mouse bite, Pin-hole
        deficit_raw = cv2.subtract(t_copper, a_copper)

        # Copper Excess (Extra Copper): Short circuit, Spur, Spurious Copper
        excess_raw = cv2.subtract(a_copper, t_copper)

        # Combined raw difference
        combined_raw = cv2.bitwise_or(deficit_raw, excess_raw)

        # Morphological opening removes thin 1-pixel boundary misregistration artifacts
        # followed by closing to bridge fragmented sub-regions belonging to the same defect
        deficit_clean = cv2.morphologyEx(
            deficit_raw, cv2.MORPH_OPEN, self.kernel_open, iterations=self.morph_iterations
        )
        deficit_clean = cv2.morphologyEx(
            deficit_clean, cv2.MORPH_CLOSE, self.kernel_close, iterations=self.morph_iterations
        )

        excess_clean = cv2.morphologyEx(
            excess_raw, cv2.MORPH_OPEN, self.kernel_open, iterations=self.morph_iterations
        )
        excess_clean = cv2.morphologyEx(
            excess_clean, cv2.MORPH_CLOSE, self.kernel_close, iterations=self.morph_iterations
        )

        combined_clean = cv2.bitwise_or(deficit_clean, excess_clean)
        diff_count = int(cv2.countNonZero(combined_clean))

        return DifferenceResult(
            copper_deficit_raw=deficit_raw,
            copper_excess_raw=excess_raw,
            combined_raw=combined_raw,
            copper_deficit_clean=deficit_clean,
            copper_excess_clean=excess_clean,
            combined_clean=combined_clean,
            diff_pixel_count=diff_count
        )
