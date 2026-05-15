"""Voxel downsampling and statistical outlier removal."""
import numpy as np
import open3d as o3d

VOXEL_SIZE_M = 0.01
OUTLIER_K    = 20
OUTLIER_STD  = 2.0


def _to_o3d(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def preprocess(
    xyz: np.ndarray,
    voxel_size: float = VOXEL_SIZE_M,
    outlier_k: int = OUTLIER_K,
    outlier_std: float = OUTLIER_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (inlier_xyz, outlier_xyz) after downsampling and denoising."""
    pcd = _to_o3d(xyz)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    all_xyz = np.asarray(pcd.points)
    _, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=outlier_k, std_ratio=outlier_std
    )
    mask = np.zeros(len(all_xyz), dtype=bool)
    mask[inlier_idx] = True
    return all_xyz[mask], all_xyz[~mask]
