"""Input Image Validator for Industrial AOI Pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


@dataclass
class ValidationResult:
    """Standardized validation outcome container."""
    is_valid: bool
    status: str  # "OK", "CORRUPTED", "BLANK", "INVALID_DIMENSIONS", "SECURITY_VIOLATION"
    message: str
    dimensions: Optional[Tuple[int, int, int]] = None
    mean_intensity: Optional[float] = None
    std_intensity: Optional[float] = None


class ImageValidator:
    """Industrial image integrity and quality pre-flight validator."""

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
    MIN_DIMENSION = 32
    MAX_DIMENSION = 8192
    MIN_CONTRAST_STD = 8.0  # Standard deviation threshold for non-blank image

    @classmethod
    def validate_file_path(cls, path: Path) -> ValidationResult:
        """Validate file path existence, extension, and file size."""
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                status="FILE_NOT_FOUND",
                message=f"File not found: {path}"
            )

        if not path.is_file():
            return ValidationResult(
                is_valid=False,
                status="NOT_A_FILE",
                message=f"Provided path is not a file: {path}"
            )

        ext = path.suffix.lower()
        if ext not in cls.ALLOWED_EXTENSIONS:
            return ValidationResult(
                is_valid=False,
                status="UNSUPPORTED_FORMAT",
                message=f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(cls.ALLOWED_EXTENSIONS))}"
            )

        size = path.stat().st_size
        if size == 0:
            return ValidationResult(
                is_valid=False,
                status="EMPTY_FILE",
                message="File size is 0 bytes."
            )

        if size > cls.MAX_FILE_SIZE_BYTES:
            return ValidationResult(
                is_valid=False,
                status="FILE_TOO_LARGE",
                message=f"File size {size / (1024*1024):.1f}MB exceeds limit of 50MB."
            )

        return ValidationResult(is_valid=True, status="OK", message="File header check passed.")

    @classmethod
    def validate_image_array(cls, image: Optional[np.ndarray], label: str = "Image") -> ValidationResult:
        """Validate decoded NumPy image array attributes and photometric quality."""
        if image is None:
            return ValidationResult(
                is_valid=False,
                status="DECODE_FAILED",
                message=f"Failed to decode {label}. File may be corrupted or not a valid image."
            )

        if not isinstance(image, np.ndarray):
            return ValidationResult(
                is_valid=False,
                status="INVALID_TYPE",
                message=f"{label} is not a valid numpy ndarray (got {type(image)})."
            )

        if image.size == 0 or len(image.shape) < 2:
            return ValidationResult(
                is_valid=False,
                status="INVALID_DIMENSIONS",
                message=f"{label} has empty or degenerate dimensions: {image.shape}."
            )

        h, w = image.shape[:2]
        c = image.shape[2] if len(image.shape) > 2 else 1

        if h < cls.MIN_DIMENSION or w < cls.MIN_DIMENSION:
            return ValidationResult(
                is_valid=False,
                status="RESOLUTION_TOO_LOW",
                message=f"{label} resolution ({w}x{h}) is below minimum requirement ({cls.MIN_DIMENSION}x{cls.MIN_DIMENSION}).",
                dimensions=(h, w, c)
            )

        if h > cls.MAX_DIMENSION or w > cls.MAX_DIMENSION:
            return ValidationResult(
                is_valid=False,
                status="RESOLUTION_EXCESSIVE",
                message=f"{label} resolution ({w}x{h}) exceeds upper limit ({cls.MAX_DIMENSION}x{cls.MAX_DIMENSION}).",
                dimensions=(h, w, c)
            )

        # Photometric validation
        gray = image if c == 1 else (image.mean(axis=2) if len(image.shape) == 3 else image)
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))

        if std_val < cls.MIN_CONTRAST_STD:
            return ValidationResult(
                is_valid=False,
                status="LOW_CONTRAST_OR_BLANK",
                message=f"{label} has negligible contrast (std dev = {std_val:.2f} < {cls.MIN_CONTRAST_STD}). Image may be solid black, white, or overexposed.",
                dimensions=(h, w, c),
                mean_intensity=mean_val,
                std_intensity=std_val
            )

        return ValidationResult(
            is_valid=True,
            status="OK",
            message=f"{label} validated successfully ({w}x{h}, {c} channels).",
            dimensions=(h, w, c),
            mean_intensity=mean_val,
            std_intensity=std_val
        )
