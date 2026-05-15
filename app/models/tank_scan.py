from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import numpy as np


class ScanType(str):
    INTERNAL = "internal"
    EXTERNAL = "external"


class Label(IntEnum):
    RUBBISH  = 0
    SHELL    = 1
    FLOOR    = 2
    DEADWOOD = 3
    ROOF     = 4


# RGB 0-255 per label — used by both viewer and exporter
LABEL_COLORS: dict[Label, tuple[int, int, int]] = {
    Label.RUBBISH:  (128, 128, 128),
    Label.SHELL:    (220,  50,  50),
    Label.FLOOR:    (220, 190,  30),
    Label.DEADWOOD: ( 50, 190,  70),
    Label.ROOF:     ( 60, 120, 220),
}

LABEL_DISPLAY_NAMES = {
    Label.RUBBISH:  "Rubbish / Noise",
    Label.SHELL:    "Shell",
    Label.FLOOR:    "Floor",
    Label.DEADWOOD: "Deadwood",
    Label.ROOF:     "Roof",
}


@dataclass
class BoundingCylinder:
    center_xy: tuple[float, float]
    radius_m: float
    base_z_m: float
    top_z_m: float

    @property
    def height_m(self) -> float:
        return self.top_z_m - self.base_z_m

    @property
    def diameter_m(self) -> float:
        return self.radius_m * 2


@dataclass
class SegmentationResult:
    points: np.ndarray             # (N, 3) float64
    labels: np.ndarray             # (N,)   int8
    scan_type: str = ScanType.EXTERNAL
    cylinder: Optional[BoundingCylinder] = None
    floor_z_m: Optional[float] = None
    roof_z_m: Optional[float] = None
    scan_metadata: dict = field(default_factory=dict)

    def point_counts(self) -> dict[str, int]:
        return {label.name: int(np.sum(self.labels == label)) for label in Label}
