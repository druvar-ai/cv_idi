"""Classical Rule-Based Defect Classifier for Bare PCB Manufacturing."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .defect_extractor import DefectCandidate


@dataclass
class ClassifiedDefect:
    """Final verified defect with class ID, name, coordinates, and confidence."""
    defect_id: int
    class_id: int             # 1: open, 2: short, 3: mousebite, 4: spur, 5: copper, 6: pin-hole
    class_name: str           # Standard DeepPCB category name
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: Tuple[float, float]
    area: float
    confidence: float
    rule_applied: str
    polarity: str


class ClassicalDefectClassifier:
    """Deterministic, rule-based defect categorizer matching DeepPCB definitions.
    
    Categories:
      1: open circuit (copper deficit breaking a conductor)
      2: short circuit (copper excess bridging conductors)
      3: mouse bite (copper deficit notch on conductor margin)
      4: spur (copper excess protrusion from conductor margin)
      5: spurious copper (isolated copper residue on substrate)
      6: pin-hole (small circular copper deficit void inside conductor)
    """

    CLASS_MAP: Dict[int, str] = {
        1: "open",
        2: "short",
        3: "mousebite",
        4: "spur",
        5: "copper",
        6: "pin-hole"
    }

    def __init__(
        self,
        pinhole_circularity_min: float = 0.52,
        pinhole_max_area: float = 380.0,
        open_aspect_ratio_min: float = 1.7,
        spur_max_area: float = 550.0
    ):
        self.pinhole_circularity_min = pinhole_circularity_min
        self.pinhole_max_area = pinhole_max_area
        self.open_aspect_ratio_min = open_aspect_ratio_min
        self.spur_max_area = spur_max_area

    def classify(self, candidate: DefectCandidate) -> ClassifiedDefect:
        """Apply deterministic classical geometric rules to candidate blob."""
        if candidate.polarity == "deficit":
            return self._classify_deficit(candidate)
        else:
            return self._classify_excess(candidate)

    def _classify_deficit(self, c: DefectCandidate) -> ClassifiedDefect:
        """Categorize copper deficits into Pin-hole (6), Open circuit (1), or Mouse bite (3)."""
        # Rule 6: Pin-hole (Class 6 in DeepPCB)
        # Void enclosed largely inside copper conductor
        if (
            (c.circularity >= self.pinhole_circularity_min or c.aspect_ratio < 1.55)
            and c.area <= self.pinhole_max_area
            and c.surrounding_copper_ratio >= 0.38
        ):
            conf = min(0.98, 0.70 + 0.30 * c.circularity)
            return ClassifiedDefect(
                defect_id=c.id,
                class_id=6,
                class_name="pin-hole",
                bbox=c.bbox,
                centroid=c.centroid,
                area=c.area,
                confidence=conf,
                rule_applied="Rule-6: Circular void enclosed in copper (Class 6 Pin-hole)",
                polarity=c.polarity
            )

        # Rule 1: Open Circuit (Class 1 in DeepPCB)
        # Significant aspect ratio or larger area cutting across trace width
        if c.aspect_ratio >= self.open_aspect_ratio_min or (c.area >= 180 and c.surrounding_copper_ratio >= 0.30):
            conf = min(0.96, 0.65 + 0.15 * min(c.aspect_ratio / 3.0, 2.0))
            return ClassifiedDefect(
                defect_id=c.id,
                class_id=1,
                class_name="open",
                bbox=c.bbox,
                centroid=c.centroid,
                area=c.area,
                confidence=conf,
                rule_applied="Rule-1: Elongated deficit cutting across conductor (Class 1 Open)",
                polarity=c.polarity
            )

        # Rule 3: Mouse Bite (Class 3 in DeepPCB)
        # Margin indentation/notch on trace perimeter
        conf = 0.88
        return ClassifiedDefect(
            defect_id=c.id,
            class_id=3,
            class_name="mousebite",
            bbox=c.bbox,
            centroid=c.centroid,
            area=c.area,
            confidence=conf,
            rule_applied="Rule-3: Conductor edge notch defect (Class 3 Mousebite)",
            polarity=c.polarity
        )

    def _classify_excess(self, c: DefectCandidate) -> ClassifiedDefect:
        """Categorize copper excesses into Short (2), Spur (4), or Spurious Copper (5)."""
        # Rule 5: Spurious Copper (Class 5 in DeepPCB)
        # Isolated copper residue on substrate (low surrounding copper ratio)
        if c.surrounding_copper_ratio < 0.22:
            conf = min(0.95, 0.75 + 0.20 * (1.0 - c.surrounding_copper_ratio))
            return ClassifiedDefect(
                defect_id=c.id,
                class_id=5,
                class_name="copper",
                bbox=c.bbox,
                centroid=c.centroid,
                area=c.area,
                confidence=conf,
                rule_applied="Rule-5: Isolated copper residue on substrate (Class 5 Spurious Copper)",
                polarity=c.polarity
            )

        # Rule 2: Short Circuit (Class 2 in DeepPCB)
        # Bridging between traces: larger area or higher aspect ratio
        if c.aspect_ratio >= 1.65 or c.area >= 200:
            conf = min(0.97, 0.70 + 0.15 * min(c.aspect_ratio / 2.5, 1.8))
            return ClassifiedDefect(
                defect_id=c.id,
                class_id=2,
                class_name="short",
                bbox=c.bbox,
                centroid=c.centroid,
                area=c.area,
                confidence=conf,
                rule_applied="Rule-2: Conductor bridging copper excess (Class 2 Short)",
                polarity=c.polarity
            )

        # Rule 4: Spur (Class 4 in DeepPCB)
        # Protrusion sticking out from a copper trace
        conf = 0.86
        return ClassifiedDefect(
            defect_id=c.id,
            class_id=4,
            class_name="spur",
            bbox=c.bbox,
            centroid=c.centroid,
            area=c.area,
            confidence=conf,
            rule_applied="Rule-4: Protrusion extending from conductor boundary (Class 4 Spur)",
            polarity=c.polarity
        )
