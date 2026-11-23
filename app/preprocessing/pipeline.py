"""Unified Preprocessing Pipeline Orchestrator."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from .enhancement import apply_clahe, apply_contrast_stretching
from .filters import apply_gaussian_filter, apply_median_filter
from .morphology import morphological_close, morphological_open
from .thresholding import apply_otsu_threshold, apply_fixed_threshold


@dataclass
class PreprocessedResult:
    """Container for preprocessed image and inspectable intermediate stages."""
    final_image: np.ndarray
    binary_mask: Optional[np.ndarray]
    stages: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PreprocessingPipeline:
    """Configurable classical computer vision preprocessing pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def process_bare_pcb(
        self,
        image_bgr: np.ndarray,
        target_size: Optional[Tuple[int, int]] = (640, 640)
    ) -> PreprocessedResult:
        """Preprocessing pipeline tailored for bare PCB fabrication images."""
        stages: Dict[str, np.ndarray] = {}

        # Stage 1: Resize if specified
        if target_size is not None and (image_bgr.shape[0] != target_size[1] or image_bgr.shape[1] != target_size[0]):
            resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
        else:
            resized = image_bgr.copy()
        stages["1_resized"] = resized

        # Stage 2: Grayscale conversion
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized.copy()
        stages["2_grayscale"] = gray

        # Stage 3: Noise filtration (Gaussian blur)
        k = self.config.get("gaussian_kernel", [5, 5])
        sigma = float(self.config.get("gaussian_sigma", 1.0))
        denoised = apply_gaussian_filter(gray, ksize=tuple(k), sigma=sigma)
        stages["3_denoised"] = denoised

        # Stage 4: Contrast enhancement (optional CLAHE)
        if self.config.get("use_clahe", False):
            enhanced = apply_clahe(denoised, clip_limit=float(self.config.get("clahe_clip_limit", 2.0)))
        else:
            enhanced = denoised
        stages["4_enhanced"] = enhanced

        # Stage 5: Binarization (Otsu or Fixed)
        bin_method = self.config.get("binarization_method", "otsu").lower()
        if bin_method == "otsu":
            thresh_val, binary = apply_otsu_threshold(enhanced)
        else:
            thresh_val = float(self.config.get("fixed_threshold", 128))
            binary = apply_fixed_threshold(enhanced, threshold=int(thresh_val))
        stages["5_binary"] = binary

        # Stage 6: Morphological cleanup (remove sub-pixel noise)
        cleaned = morphological_open(binary, kernel_size=3, shape="rect")
        stages["6_morph_cleaned"] = cleaned

        return PreprocessedResult(
            final_image=cleaned,
            binary_mask=cleaned,
            stages=stages,
            metadata={"binarization_method": bin_method, "threshold_value": thresh_val}
        )

    def process_pcba(
        self,
        image_bgr: np.ndarray,
        resize_factor: float = 0.5
    ) -> PreprocessedResult:
        """Preprocessing pipeline tailored for high-resolution assembled PCB images."""
        stages: Dict[str, np.ndarray] = {}

        # Stage 1: Optional downsampling for performance
        if resize_factor != 1.0:
            h, w = image_bgr.shape[:2]
            new_w, new_h = int(w * resize_factor), int(h * resize_factor)
            resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = image_bgr.copy()
        stages["1_resized"] = resized

        # Stage 2: Color enhancement via CLAHE in LAB space
        enhanced_bgr = apply_clahe(resized, clip_limit=2.5, tile_grid_size=(8, 8))
        stages["2_clahe_bgr"] = enhanced_bgr

        # Stage 3: Grayscale conversion
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        stages["3_grayscale"] = gray

        # Stage 4: Edge-preserving median / Gaussian smoothing
        smoothed = apply_gaussian_filter(gray, ksize=(5, 5), sigma=1.0)
        stages["4_smoothed"] = smoothed

        # Stage 5: Canny edge map for component boundary inspection
        edges = cv2.Canny(smoothed, threshold1=50, threshold2=150)
        stages["5_edges"] = edges

        return PreprocessedResult(
            final_image=smoothed,
            binary_mask=edges,
            stages=stages,
            metadata={"resize_factor": resize_factor}
        )
