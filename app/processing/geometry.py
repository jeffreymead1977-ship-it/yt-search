"""RANSAC plane and vertical cylinder fitting."""
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from typing import Optional

RANSAC_ITERATIONS        = 1000
PLANE_DISTANCE_THRESHOLD = 0.015   # 15 mm
CYLINDER_RADIAL_TOLERANCE = 0.020  # 20 mm


@dataclass
class PlaneModel:
    normal: np.ndarray    # unit normal (a, b, c)
    d: float              # plane: dot(normal, p) + d = 0
    inlier_mask: np.ndarray

    def distance(self, pts: np.ndarray) -> np.ndarray:
        return np.abs(pts @ self.normal + self.d)


@dataclass
class CylinderModel:
    axis_point: np.ndarray   # point on the (vertical) axis
    radius: float
    inlier_mask: np.ndarray


def fit_plane_ransac(
    xyz: np.ndarray,
    distance_threshold: float = PLANE_DISTANCE_THRESHOLD,
    n_iterations: int = RANSAC_ITERATIONS,
) -> Optional[PlaneModel]:
    if len(xyz) < 3:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    plane_eq, inlier_idx = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=n_iterations,
    )
    a, b, c, d = plane_eq
    norm = np.linalg.norm([a, b, c])
    normal = np.array([a, b, c]) / norm
    mask = np.zeros(len(xyz), dtype=bool)
    mask[inlier_idx] = True
    return PlaneModel(normal=normal, d=d / norm, inlier_mask=mask)


def _circle_from_points(pts: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
    """Algebraic circle fit in XY plane (Taubin). Returns (center_xy, radius) or None."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    try:
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = res[0] / 2, res[1] / 2
    r = np.sqrt(max(res[2] + cx ** 2 + cy ** 2, 0))
    if r <= 0 or r > 200:
        return None
    return np.array([cx, cy]), r


def fit_cylinder_ransac(
    xyz: np.ndarray,
    radial_tolerance: float = CYLINDER_RADIAL_TOLERANCE,
    n_iterations: int = RANSAC_ITERATIONS,
    min_inlier_fraction: float = 0.3,
) -> Optional[CylinderModel]:
    """RANSAC vertical cylinder fit (axis assumed ≈ Z)."""
    if len(xyz) < 10:
        return None

    rng = np.random.default_rng(42)
    best_mask: Optional[np.ndarray] = None
    best_count = 0
    best_center = np.zeros(2)
    best_radius = 0.0

    for _ in range(n_iterations):
        idx = rng.choice(len(xyz), size=5, replace=False)
        result = _circle_from_points(xyz[idx])
        if result is None:
            continue
        center_xy, radius = result
        dx = xyz[:, 0] - center_xy[0]
        dy = xyz[:, 1] - center_xy[1]
        dist = np.abs(np.sqrt(dx ** 2 + dy ** 2) - radius)
        mask = dist < radial_tolerance
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
            best_center, best_radius = center_xy, radius

    if best_mask is None or best_count < min_inlier_fraction * len(xyz):
        return None

    # Refine on inliers
    refined = _circle_from_points(xyz[best_mask])
    if refined is not None:
        best_center, best_radius = refined
        dx = xyz[:, 0] - best_center[0]
        dy = xyz[:, 1] - best_center[1]
        best_mask = np.abs(np.sqrt(dx ** 2 + dy ** 2) - best_radius) < radial_tolerance

    return CylinderModel(
        axis_point=np.array([best_center[0], best_center[1], 0.0]),
        radius=best_radius,
        inlier_mask=best_mask,
    )
