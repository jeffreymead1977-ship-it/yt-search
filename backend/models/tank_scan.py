from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import numpy as np


class Label(IntEnum):
    RUBBISH = 0
    SHELL = 1
    FLOOR = 2
    DEADWOOD = 3
    ROOF = 4


LABEL_COLORS = {
    Label.RUBBISH:  (0.5, 0.5, 0.5),
    Label.SHELL:    (0.9, 0.2, 0.2),
    Label.FLOOR:    (0.9, 0.8, 0.1),
    Label.ROOF:     (0.2, 0.4, 0.9),
    Label.DEADWOOD: (0.2, 0.8, 0.3),
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


@dataclass
class SegmentationResult:
    points: np.ndarray          # (N, 3) float64
    labels: np.ndarray          # (N,)   int8
    cylinder: Optional[BoundingCylinder] = None
    floor_z_m: Optional[float] = None
    roof_z_m: Optional[float] = None
    scan_metadata: dict = field(default_factory=dict)

    def point_counts(self) -> dict[str, int]:
        return {label.name: int(np.sum(self.labels == label)) for label in Label}
