"""Measurement algorithms for point cloud inspection."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MeasurementResult:
    kind:           str               # "distance" | "diameter" | "area"
    value:          float             # metres or m²
    unit:           str               # "m" or "m²"
    label:          str               # e.g. "Ø 0.324 m"
    annotation_pts: list              # world-space np.ndarray points
    extra:          dict = field(default_factory=dict)


def measure_distance(pt_a: np.ndarray, pt_b: np.ndarray) -> MeasurementResult:
    dist = float(np.linalg.norm(pt_b - pt_a))
    return MeasurementResult(
        kind="distance",
        value=dist,
        unit="m",
        label=f"↔ {dist:.3f} m",
        annotation_pts=[pt_a, pt_b],
        extra={},
    )


def measure_diameter(pts: np.ndarray) -> MeasurementResult:
    from processing.geometry import fit_cylinder_ransac

    cyl = fit_cylinder_ransac(pts)
    if cyl is not None:
        diam    = cyl.radius * 2.0
        z_vals  = pts[cyl.inlier_mask, 2]
        z_lo    = float(z_vals.min())
        z_hi    = float(z_vals.max())
        height  = z_hi - z_lo
        cx, cy  = cyl.axis_point[0], cyl.axis_point[1]
        base_pt = np.array([cx, cy, z_lo])
        top_pt  = np.array([cx, cy, z_hi])
        inlier_frac = float(cyl.inlier_mask.sum()) / max(len(pts), 1)
        return MeasurementResult(
            kind="diameter",
            value=diam,
            unit="m",
            label=f"Ø {diam:.3f} m",
            annotation_pts=[base_pt, top_pt],
            extra={
                "radius_m": cyl.radius,
                "height_m": height,
                "inlier_fraction": inlier_frac,
            },
        )

    # Fallback: bounding-box diagonal in XY
    lo  = pts.min(axis=0)
    hi  = pts.max(axis=0)
    diam = float(np.linalg.norm(hi[:2] - lo[:2]))
    mid_z = float((lo[2] + hi[2]) / 2)
    cx = float((lo[0] + hi[0]) / 2)
    cy = float((lo[1] + hi[1]) / 2)
    return MeasurementResult(
        kind="diameter",
        value=diam,
        unit="m",
        label=f"Ø {diam:.3f} m (bbox)",
        annotation_pts=[np.array([cx, cy, lo[2]]), np.array([cx, cy, hi[2]])],
        extra={"radius_m": diam / 2, "height_m": hi[2] - lo[2], "inlier_fraction": 0.0},
    )


def measure_area(pts: np.ndarray) -> MeasurementResult:
    from processing.geometry import fit_plane_ransac
    try:
        from scipy.spatial import ConvexHull
        _SCIPY_OK = True
    except ImportError:
        _SCIPY_OK = False

    plane = fit_plane_ransac(pts)
    if plane is None:
        # Degenerate fallback
        return MeasurementResult(
            kind="area", value=0.0, unit="m²",
            label="Area 0.000 m²",
            annotation_pts=list(pts[:3]),
            extra={"plane_normal": np.array([0.0, 0.0, 1.0]), "n_inliers": 0},
        )

    normal   = plane.normal
    inliers  = pts[plane.inlier_mask]
    n_inliers = int(plane.inlier_mask.sum())

    if not _SCIPY_OK or len(inliers) < 3:
        area = 0.0
        hull_3d: list[np.ndarray] = list(inliers[:3])
    else:
        # Build orthonormal basis on the plane
        n    = normal / np.linalg.norm(normal)
        ref  = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(n, ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        u = np.cross(n, ref); u /= np.linalg.norm(u)
        v = np.cross(n, u);   v /= np.linalg.norm(v)

        # Project inliers onto plane
        origin  = inliers.mean(axis=0)
        rel     = inliers - origin
        coords2d = np.column_stack([rel @ u, rel @ v])

        try:
            hull = ConvexHull(coords2d)
            area = float(hull.volume)  # ConvexHull.volume = area in 2D
            hull_verts_2d = coords2d[hull.vertices]
            hull_3d = [origin + p[0]*u + p[1]*v for p in hull_verts_2d]
        except Exception:
            area    = 0.0
            hull_3d = list(inliers[:3])

    return MeasurementResult(
        kind="area",
        value=area,
        unit="m²",
        label=f"Area {area:.3f} m²",
        annotation_pts=hull_3d,
        extra={"plane_normal": normal, "n_inliers": n_inliers},
    )
