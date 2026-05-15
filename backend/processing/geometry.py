"""Geometry primitives: RANSAC plane and cylinder fitting."""
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from typing import Optional


RANSAC_ITERATIONS = 1000
PLANE_DISTANCE_THRESHOLD = 0.015   # 15 mm
CYLINDER_RADIAL_TOLERANCE = 0.020  # 20 mm


@dataclass
class PlaneModel:
    normal: np.ndarray   # unit normal (a, b, c)
    d: float             # plane: a*x + b*y + c*z + d = 0
    inlier_mask: np.ndarray

    def distance(self, pts: np.ndarray) -> np.ndarray:
        return np.abs(pts @ self.normal + self.d)


@dataclass
class CylinderModel:
    axis: np.ndarray        # unit direction of cylinder axis (near [0,0,1])
    axis_point: np.ndarray  # a point on the axis
    radius: float
    inlier_mask: np.ndarray

    def radial_distance(self, pts: np.ndarray) -> np.ndarray:
        """Distance from each point to the cylinder surface."""
        v = pts - self.axis_point
        proj = (v @ self.axis)[:, None] * self.axis
        perp = v - proj
        return np.abs(np.linalg.norm(perp, axis=1) - self.radius)


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
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    mask = np.zeros(len(xyz), dtype=bool)
    mask[inlier_idx] = True
    return PlaneModel(normal=normal, d=d / np.linalg.norm([a, b, c]), inlier_mask=mask)


def _fit_cylinder_to_subset(pts: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
    """Fit a vertical cylinder to a small subset. Returns (center_xy, radius) or None."""
    if len(pts) < 3:
        return None
    # Project onto XY plane and fit circle via algebraic method (Taubin)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b_vec = x**2 + y**2
    try:
        result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx = result[0] / 2
    cy = result[1] / 2
    r = np.sqrt(result[2] + cx**2 + cy**2)
    if r <= 0 or r > 100:  # sanity: radius between 0 and 100 m
        return None
    return np.array([cx, cy]), r


def fit_cylinder_ransac(
    xyz: np.ndarray,
    radial_tolerance: float = CYLINDER_RADIAL_TOLERANCE,
    n_iterations: int = RANSAC_ITERATIONS,
    min_inlier_fraction: float = 0.3,
) -> Optional[CylinderModel]:
    """RANSAC cylinder fit assuming near-vertical axis."""
    if len(xyz) < 10:
        return None

    rng = np.random.default_rng(42)
    best_mask: Optional[np.ndarray] = None
    best_count = 0
    best_center = None
    best_radius = 0.0

    for _ in range(n_iterations):
        idx = rng.choice(len(xyz), size=5, replace=False)
        result = _fit_cylinder_to_subset(xyz[idx])
        if result is None:
            continue
        center_xy, radius = result

        dx = xyz[:, 0] - center_xy[0]
        dy = xyz[:, 1] - center_xy[1]
        dist_to_surface = np.abs(np.sqrt(dx**2 + dy**2) - radius)
        mask = dist_to_surface < radial_tolerance
        count = int(mask.sum())

        if count > best_count:
            best_count = count
            best_mask = mask
            best_center = center_xy
            best_radius = radius

    if best_mask is None or best_count < min_inlier_fraction * len(xyz):
        return None

    # Refine using all inliers
    result = _fit_cylinder_to_subset(xyz[best_mask])
    if result is not None:
        best_center, best_radius = result
        dx = xyz[:, 0] - best_center[0]
        dy = xyz[:, 1] - best_center[1]
        dist_to_surface = np.abs(np.sqrt(dx**2 + dy**2) - best_radius)
        best_mask = dist_to_surface < radial_tolerance

    axis = np.array([0.0, 0.0, 1.0])
    axis_point = np.array([best_center[0], best_center[1], 0.0])

    return CylinderModel(
        axis=axis,
        axis_point=axis_point,
        radius=best_radius,
        inlier_mask=best_mask,
    )
