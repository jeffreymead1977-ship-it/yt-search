"""Voxel downsampling and statistical outlier removal."""
import numpy as np
import open3d as o3d


# Tunable defaults
VOXEL_SIZE_M = 0.01       # 10 mm
OUTLIER_K = 20
OUTLIER_STD = 2.0


def to_o3d(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def from_o3d(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(pcd.points)


def preprocess(
    xyz: np.ndarray,
    voxel_size: float = VOXEL_SIZE_M,
    outlier_k: int = OUTLIER_K,
    outlier_std: float = OUTLIER_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (inlier_xyz, outlier_xyz) after downsampling and denoising."""
    pcd = to_o3d(xyz)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    all_xyz = from_o3d(pcd)

    _, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=outlier_k, std_ratio=outlier_std
    )
    inlier_mask = np.zeros(len(all_xyz), dtype=bool)
    inlier_mask[inlier_idx] = True

    return all_xyz[inlier_mask], all_xyz[~inlier_mask]
