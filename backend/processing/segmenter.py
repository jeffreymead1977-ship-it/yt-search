"""
Main segmentation pipeline.

Steps:
  1. Preprocess (downsample + outlier removal) → RUBBISH
  2. Detect floor plane (lower quartile, near-horizontal) → FLOOR
  3. Detect shell cylinder (vertical RANSAC) → SHELL
  4. Detect roof plane (upper quartile, near-horizontal) → ROOF
  5. Inside cylinder, unlabelled → DEADWOOD
  6. Outside cylinder, unlabelled → RUBBISH
"""
import numpy as np
from models.tank_scan import Label, SegmentationResult, BoundingCylinder
from processing.preprocessor import preprocess
from processing.geometry import (
    fit_plane_ransac,
    fit_cylinder_ransac,
    PLANE_DISTANCE_THRESHOLD,
    CYLINDER_RADIAL_TOLERANCE,
)

FLOOR_Z_PERCENTILE = 25
ROOF_Z_PERCENTILE = 75
VERTICAL_NORMAL_THRESHOLD = 0.94   # cos(20°) ≈ 0.94


def segment(
    xyz: np.ndarray,
    voxel_size: float = 0.01,
    plane_dist: float = PLANE_DISTANCE_THRESHOLD,
    cyl_tol: float = CYLINDER_RADIAL_TOLERANCE,
) -> SegmentationResult:

    # ── Step 1: Preprocess ──────────────────────────────────────────────────
    inliers, noise = preprocess(xyz, voxel_size=voxel_size)
    all_pts = np.concatenate([inliers, noise], axis=0)
    labels = np.full(len(all_pts), Label.RUBBISH, dtype=np.int8)
    n_inliers = len(inliers)
    unlabelled = np.ones(len(all_pts), dtype=bool)
    unlabelled[n_inliers:] = False   # noise points stay RUBBISH

    working = all_pts

    # ── Step 2: Floor ───────────────────────────────────────────────────────
    z_low_thresh = np.percentile(working[unlabelled, 2], FLOOR_Z_PERCENTILE)
    low_mask = unlabelled & (working[:, 2] <= z_low_thresh)
    floor_model = None
    floor_z = None

    if low_mask.sum() > 10:
        floor_model = fit_plane_ransac(working[low_mask], distance_threshold=plane_dist)
        if floor_model is not None:
            if abs(floor_model.normal[2]) >= VERTICAL_NORMAL_THRESHOLD:
                # Map inliers back to full index space
                low_indices = np.where(low_mask)[0]
                inlier_global = low_indices[floor_model.inlier_mask]
                labels[inlier_global] = Label.FLOOR
                unlabelled[inlier_global] = False
                floor_z = float(np.median(working[inlier_global, 2]))

    # ── Step 3: Shell ───────────────────────────────────────────────────────
    shell_pts = working[unlabelled]
    cylinder = fit_cylinder_ransac(shell_pts, radial_tolerance=cyl_tol)
    bounding_cylinder = None
    cx, cy, radius = 0.0, 0.0, 0.0

    if cylinder is not None:
        unlabelled_indices = np.where(unlabelled)[0]
        shell_global = unlabelled_indices[cylinder.inlier_mask]
        labels[shell_global] = Label.SHELL
        unlabelled[shell_global] = False
        cx, cy = cylinder.axis_point[0], cylinder.axis_point[1]
        radius = cylinder.radius

    # ── Step 4: Roof ────────────────────────────────────────────────────────
    z_high_thresh = np.percentile(working[unlabelled, 2], 100 - (100 - ROOF_Z_PERCENTILE))
    high_mask = unlabelled & (working[:, 2] >= z_high_thresh)
    roof_z = None

    if high_mask.sum() > 10:
        roof_model = fit_plane_ransac(working[high_mask], distance_threshold=plane_dist)
        if roof_model is not None:
            if abs(roof_model.normal[2]) >= VERTICAL_NORMAL_THRESHOLD:
                high_indices = np.where(high_mask)[0]
                inlier_global = high_indices[roof_model.inlier_mask]
                labels[inlier_global] = Label.ROOF
                unlabelled[inlier_global] = False
                roof_z = float(np.median(working[inlier_global, 2]))

    # ── Step 5 & 6: Deadwood vs external rubbish ───────────────────────────
    if cylinder is not None and unlabelled.any():
        remaining_idx = np.where(unlabelled)[0]
        pts_rem = working[remaining_idx]
        dx = pts_rem[:, 0] - cx
        dy = pts_rem[:, 1] - cy
        radial_dist = np.sqrt(dx**2 + dy**2)

        z_min = floor_z if floor_z is not None else working[:, 2].min()
        z_max = roof_z if roof_z is not None else working[:, 2].max()

        inside = (
            (radial_dist < radius + cyl_tol * 2) &
            (pts_rem[:, 2] >= z_min - 0.1) &
            (pts_rem[:, 2] <= z_max + 0.1)
        )
        labels[remaining_idx[inside]] = Label.DEADWOOD
        # outside stays RUBBISH

    # ── Build result ────────────────────────────────────────────────────────
    if cylinder is not None:
        base_z = floor_z if floor_z is not None else float(working[:, 2].min())
        top_z = roof_z if roof_z is not None else float(working[:, 2].max())
        bounding_cylinder = BoundingCylinder(
            center_xy=(cx, cy),
            radius_m=radius,
            base_z_m=base_z,
            top_z_m=top_z,
        )

    return SegmentationResult(
        points=all_pts,
        labels=labels,
        cylinder=bounding_cylinder,
        floor_z_m=floor_z,
        roof_z_m=roof_z,
    )
