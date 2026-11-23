"""Centralized Configuration Manager for PCB Inspection System."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import yaml


@dataclass
class SystemConfig:
    app_name: str = "Intelligent PCB Defect Inspection System"
    version: str = "1.0.0"
    phase: str = "Phase 1 - Classical Computer Vision"
    debug: bool = False
    log_level: str = "INFO"
    random_seed: int = 42


@dataclass
class PathsConfig:
    bare_pcb_dir: Path = Path("datasets/bare_pcb")
    assembled_pcb_dir: Path = Path("datasets/assembled_pcb")
    output_intermediate_dir: Path = Path("outputs/intermediate")
    output_reports_dir: Path = Path("outputs/reports")
    output_history_dir: Path = Path("outputs/history")


@dataclass
class DecisionEngineConfig:
    review_confidence_threshold: float = 0.65
    min_contrast_std: float = 15.0
    min_registration_inlier_ratio: float = 0.40
    max_alignment_error_px: float = 25.0


@dataclass
class MasterConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    decision_engine: DecisionEngineConfig = field(default_factory=DecisionEngineConfig)
    bare_pcb: Dict[str, Any] = field(default_factory=dict)
    pcba: Dict[str, Any] = field(default_factory=dict)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Safely load a YAML configuration file."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config(
    default_config_path: str = "configs/default_config.yaml",
    bare_config_path: str = "configs/bare_pcb_config.yaml",
    pcba_config_path: str = "configs/pcba_config.yaml",
) -> MasterConfig:
    """Load and assemble the master configuration."""
    base_dict = load_yaml(Path(default_config_path))
    bare_dict = load_yaml(Path(bare_config_path))
    pcba_dict = load_yaml(Path(pcba_config_path))

    sys_raw = base_dict.get("system", {})
    paths_raw = base_dict.get("paths", {})
    de_raw = base_dict.get("decision_engine", {})

    cfg = MasterConfig(
        system=SystemConfig(
            app_name=sys_raw.get("app_name", "Intelligent PCB Defect Inspection System"),
            version=sys_raw.get("version", "1.0.0"),
            phase=sys_raw.get("phase", "Phase 1 - Classical Computer Vision"),
            debug=sys_raw.get("debug", False),
            log_level=sys_raw.get("log_level", "INFO"),
            random_seed=sys_raw.get("random_seed", 42),
        ),
        paths=PathsConfig(
            bare_pcb_dir=Path(paths_raw.get("bare_pcb_dir", "datasets/bare_pcb")),
            assembled_pcb_dir=Path(paths_raw.get("assembled_pcb_dir", "datasets/assembled_pcb")),
            output_intermediate_dir=Path(paths_raw.get("output_intermediate_dir", "outputs/intermediate")),
            output_reports_dir=Path(paths_raw.get("output_reports_dir", "outputs/reports")),
            output_history_dir=Path(paths_raw.get("output_history_dir", "outputs/history")),
        ),
        decision_engine=DecisionEngineConfig(
            review_confidence_threshold=float(de_raw.get("review_confidence_threshold", 0.65)),
            min_contrast_std=float(de_raw.get("min_contrast_std", 15.0)),
            min_registration_inlier_ratio=float(de_raw.get("min_registration_inlier_ratio", 0.40)),
            max_alignment_error_px=float(de_raw.get("max_alignment_error_px", 25.0)),
        ),
        bare_pcb=bare_dict,
        pcba=pcba_dict,
    )
    return cfg
