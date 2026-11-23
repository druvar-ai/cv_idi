"""Safe Image Loader for File Paths, Streams, and Uploaded Buffers."""

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union
import cv2
import numpy as np

from .validator import ImageValidator, ValidationResult


@dataclass
class LoadedImage:
    """Encapsulates image data with metadata and validation state."""
    image_bgr: np.ndarray
    image_rgb: np.ndarray
    image_gray: np.ndarray
    dimensions: Tuple[int, int, int]  # (height, width, channels)
    source_name: str
    validation: ValidationResult


class ImageLoader:
    """Industrial grade image ingestion handler."""

    @classmethod
    def load_from_path(cls, file_path: Union[str, Path]) -> Tuple[Optional[LoadedImage], ValidationResult]:
        """Load an image from a local file path with full safety checks."""
        path = Path(file_path)
        val_path = ImageValidator.validate_file_path(path)
        if not val_path.is_valid:
            return None, val_path

        # Read using OpenCV (handles BGR order)
        # Using cv2.imdecode to avoid path encoding issues on Windows
        try:
            with open(path, "rb") as f:
                data = f.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            return None, ValidationResult(
                is_valid=False,
                status="READ_ERROR",
                message=f"Failed to read file from disk: {str(e)}"
            )

        val_arr = ImageValidator.validate_image_array(img_bgr, label=path.name)
        if not val_arr.is_valid:
            return None, val_arr

        return cls._create_loaded_image(img_bgr, source_name=path.name, val_result=val_arr), val_arr

    @classmethod
    def load_from_bytes(cls, raw_bytes: bytes, source_name: str = "upload") -> Tuple[Optional[LoadedImage], ValidationResult]:
        """Load an image from memory bytes (e.g. Streamlit FileUploader)."""
        if not raw_bytes or len(raw_bytes) == 0:
            res = ValidationResult(is_valid=False, status="EMPTY_BYTES", message="Provided bytes buffer is empty.")
            return None, res

        try:
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            res = ValidationResult(is_valid=False, status="DECODE_EXCEPTION", message=f"Decode error: {str(e)}")
            return None, res

        val_arr = ImageValidator.validate_image_array(img_bgr, label=source_name)
        if not val_arr.is_valid:
            return None, val_arr

        return cls._create_loaded_image(img_bgr, source_name=source_name, val_result=val_arr), val_arr

    @staticmethod
    def _create_loaded_image(img_bgr: np.ndarray, source_name: str, val_result: ValidationResult) -> LoadedImage:
        """Create standard multi-color-space container."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = img_bgr.shape[:2]
        c = img_bgr.shape[2] if len(img_bgr.shape) > 2 else 1
        return LoadedImage(
            image_bgr=img_bgr,
            image_rgb=img_rgb,
            image_gray=img_gray,
            dimensions=(h, w, c),
            source_name=source_name,
            validation=val_result
        )
