"""
Six-step segmentation pipeline:
  1. Preprocess → noise becomes RUBBISH
  2. Fit floor plane (lower quartile, near-horizontal)
  3. Fit shell cylinder (vertical RANSAC)
  4. Fit roof plane (upper quartile, near-horizontal)
  5. Inside cylinder, unlabelled → DEADWOOD
  6. Outside cylinder, unlabelled → RUBBISH
"""
import numpy as np
from models.tank_scan import Label, SegmentationResult, BoundingCylinder, ScanType
from processing.preprocessor import preprocess
from processing.geometry import (
    fit_plane_ransac,
    fit_cylinder_ransac,
    PLANE_DISTANCE_THRESHOLD,
    CYLINDER_RADIAL_TOLERANCE,
)

FLOOR_Z_PERCENTILE      = 25
ROOF_Z_PERCENTILE       = 75
VERTICAL_NORMAL_MIN     = 0.94    # cos(20°)


def segment(
    xyz: np.ndarray,
    scan_type: str = ScanType.EXTERNAL,
    voxel_size: float = 0.01,
    plane_dist: float = PLANE_DISTANCE_THRESHOLD,
    cyl_tol: float = CYLINDER_RADIAL_TOLERANCE,
    progress_cb=None,
) -> SegmentationResult:

    def _progress(msg: str):
        if progress_cb:
            progress_cb(msg)

    # ── 1. Preprocess ────────────────────────────────────────────────────────
    _progress("Preprocessing: downsampling and removing noise…")
    inliers, noise = preprocess(xyz, voxel_size=voxel_size)
    all_pts = np.concatenate([inliers, noise], axis=0)
    labels = np.full(len(all_pts), Label.RUBBISH, dtype=np.int8)
    unlabelled = np.zeros(len(all_pts), dtype=bool)
    unlabelled[:len(inliers)] = True

    # ── 2. Floor ────────────────────────────────────────────────────────────
    _progress("Fitting floor plane…")
    z_low = np.percentile(all_pts[unlabelled, 2], FLOOR_Z_PERCENTILE)
    low_mask = unlabelled & (all_pts[:, 2] <= z_low)
    floor_z = None

    if low_mask.sum() > 10:
        floor_model = fit_plane_ransac(all_pts[low_mask], distance_threshold=plane_dist)
        if floor_model is not None and abs(floor_model.normal[2]) >= VERTICAL_NORMAL_MIN:
            low_idx = np.where(low_mask)[0]
            hit = low_idx[floor_model.inlier_mask]
            labels[hit] = Label.FLOOR
            unlabelled[hit] = False
            floor_z = float(np.median(all_pts[hit, 2]))

    # ── 3. Shell ────────────────────────────────────────────────────────────
    _progress("Fitting shell cylinder…")
    cylinder = fit_cylinder_ransac(all_pts[unlabelled], radial_tolerance=cyl_tol)
    bounding_cyl = None
    cx = cy = radius = 0.0

    if cylinder is not None:
        ul_idx = np.where(unlabelled)[0]
        hit = ul_idx[cylinder.inlier_mask]
        labels[hit] = Label.SHELL
        unlabelled[hit] = False
        cx, cy = cylinder.axis_point[0], cylinder.axis_point[1]
        radius = cylinder.radius

    # ── 4. Roof ─────────────────────────────────────────────────────────────
    _progress("Fitting roof plane…")
    z_high = np.percentile(all_pts[unlabelled, 2], ROOF_Z_PERCENTILE)
    high_mask = unlabelled & (all_pts[:, 2] >= z_high)
    roof_z = None

    if high_mask.sum() > 10:
        roof_model = fit_plane_ransac(all_pts[high_mask], distance_threshold=plane_dist)
        if roof_model is not None and abs(roof_model.normal[2]) >= VERTICAL_NORMAL_MIN:
            high_idx = np.where(high_mask)[0]
            hit = high_idx[roof_model.inlier_mask]
            labels[hit] = Label.ROOF
            unlabelled[hit] = False
            roof_z = float(np.median(all_pts[hit, 2]))

    # ── 5 & 6. Deadwood vs external rubbish ─────────────────────────────────
    _progress("Classifying internal structure…")
    if cylinder is not None and unlabelled.any():
        rem_idx = np.where(unlabelled)[0]
        pts_r = all_pts[rem_idx]
        dx = pts_r[:, 0] - cx
        dy = pts_r[:, 1] - cy
        radial = np.sqrt(dx ** 2 + dy ** 2)
        z_min = (floor_z - 0.1) if floor_z is not None else all_pts[:, 2].min()
        z_max = (roof_z + 0.1) if roof_z is not None else all_pts[:, 2].max()
        inside = (radial < radius + cyl_tol * 2) & (pts_r[:, 2] >= z_min) & (pts_r[:, 2] <= z_max)
        labels[rem_idx[inside]] = Label.DEADWOOD

    # ── Build result ─────────────────────────────────────────────────────────
    if cylinder is not None:
        base_z = floor_z if floor_z is not None else float(all_pts[:, 2].min())
        top_z  = roof_z  if roof_z  is not None else float(all_pts[:, 2].max())
        bounding_cyl = BoundingCylinder(
            center_xy=(cx, cy),
            radius_m=radius,
            base_z_m=base_z,
            top_z_m=top_z,
        )

    _progress("Done.")
    return SegmentationResult(
        points=all_pts,
        labels=labels,
        scan_type=scan_type,
        cylinder=bounding_cyl,
        floor_z_m=floor_z,
        roof_z_m=roof_z,
    )
