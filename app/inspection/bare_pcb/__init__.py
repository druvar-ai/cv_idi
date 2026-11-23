"""Bare PCB Fabrication Inspection Subsystem Package."""

from .classifier import ClassicalDefectClassifier, ClassifiedDefect
from .defect_extractor import DefectCandidate, DefectExtractor
from .diff_analyzer import DifferenceAnalyzer, DifferenceResult
from .pipeline import BarePCBInspectionPipeline, BarePCBInspectionResult

__all__ = [
    "ClassicalDefectClassifier",
    "ClassifiedDefect",
    "DefectCandidate",
    "DefectExtractor",
    "DifferenceAnalyzer",
    "DifferenceResult",
    "BarePCBInspectionPipeline",
    "BarePCBInspectionResult",
]
