"""Component Region Extraction and Specification Loader for PCBA AOI."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import cv2
import numpy as np


@dataclass
class ComponentROI:
    """Component region of interest and nominal placement specification."""
    name: str
    component_type: str
    bbox: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax) in reference template space
    nominal_center: Tuple[float, float]
    width: int
    height: int
    nominal_rotation_deg: float = 0.0


class ComponentSpecificationLoader:
    """Loads nominal component placement data from Pascal VOC XML, JSON, or visual saliency."""

    @classmethod
    def load_from_xml(cls, xml_path: Union[str, Path]) -> List[ComponentROI]:
        """Parse component bounding boxes and identities from Pascal VOC XML."""
        p = Path(xml_path)
        if not p.exists():
            return []

        tree = ET.parse(p)
        root = tree.getroot()
        components: List[ComponentROI] = []

        for obj in root.findall("object"):
            name = obj.find("name").text or "component"
            bnd = obj.find("bndbox")
            if bnd is None:
                continue

            xmin = int(float(bnd.find("xmin").text))
            ymin = int(float(bnd.find("ymin").text))
            xmax = int(float(bnd.find("xmax").text))
            ymax = int(float(bnd.find("ymax").text))

            w = max(1, xmax - xmin)
            h = max(1, ymax - ymin)
            cx = xmin + w / 2.0
            cy = ymin + h / 2.0

            # Derive component type code (e.g. '1_a' -> type 'a')
            parts = name.split("_")
            c_type = parts[1] if len(parts) > 1 else "generic"

            components.append(
                ComponentROI(
                    name=name,
                    component_type=c_type,
                    bbox=(xmin, ymin, xmax, ymax),
                    nominal_center=(cx, cy),
                    width=w,
                    height=h
                )
            )

        return components

    @classmethod
    def auto_segment_components(
        cls,
        template_gray: np.ndarray,
        min_area: int = 400,
        max_area: int = 50000
    ) -> List[ComponentROI]:
        """Auto-discover high-contrast component clusters using morphological saliency."""
        # Multi-scale edge and morphological blackhat to locate dark IC packages
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(template_gray, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(blackhat, 40, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        discovered: List[ComponentROI] = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                name = f"auto_comp_{i}"
                discovered.append(
                    ComponentROI(
                        name=name,
                        component_type="auto",
                        bbox=(x, y, x + w, y + h),
                        nominal_center=(x + w / 2.0, y + h / 2.0),
                        width=w,
                        height=h
                    )
                )

        return discovered
