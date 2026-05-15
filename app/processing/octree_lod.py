"""Spatially-adaptive octree LOD (Potree-style).

Each node owns a bounding sphere. At render time nodes whose projected sphere
radius is smaller than threshold_px render their own coarse sample; larger
nodes recurse into children — close geometry = fine detail, far = coarse.
GPU budget stays at ~2-4 M points regardless of cloud size.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class OctreeNode:
    center:   np.ndarray          # (3,) float64 — bounding-sphere centre
    half_diag: float              # bounding-sphere radius (half space diagonal)
    sample:   np.ndarray          # int indices into the original pts array
    children: list                # list of 8 OctreeNode | None
    is_leaf:  bool


def build_octree(
    pts: np.ndarray,
    max_depth: int = 8,
    min_pts: int = 500,
    samples_per_node: int = 150,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> OctreeNode:
    """Build an octree over pts and return the root OctreeNode."""
    rng = np.random.default_rng(0)
    total = [0]

    def _build(indices: np.ndarray, depth: int) -> OctreeNode:
        sub = pts[indices]
        lo  = sub.min(axis=0)
        hi  = sub.max(axis=0)
        center    = (lo + hi) * 0.5
        half_diag = float(np.linalg.norm(hi - lo) * 0.5)

        n = len(indices)
        k = min(samples_per_node, n)
        sample = rng.choice(indices, k, replace=False)

        if depth >= max_depth or n <= min_pts:
            node = OctreeNode(center=center, half_diag=half_diag,
                              sample=sample, children=[None] * 8, is_leaf=True)
            total[0] += n
            if progress_cb and total[0] % 200_000 < n:
                progress_cb(f"Building octree… {total[0]:,} pts indexed")
            return node

        mid = center
        children: list = []
        for oct_i in range(8):
            dx = 1 if (oct_i & 1) else 0
            dy = 1 if (oct_i & 2) else 0
            dz = 1 if (oct_i & 4) else 0
            x_lo, x_hi = (lo[0], mid[0]) if dx == 0 else (mid[0], hi[0])
            y_lo, y_hi = (lo[1], mid[1]) if dy == 0 else (mid[1], hi[1])
            z_lo, z_hi = (lo[2], mid[2]) if dz == 0 else (mid[2], hi[2])
            mask = (
                (sub[:, 0] >= x_lo) & (sub[:, 0] < x_hi) &
                (sub[:, 1] >= y_lo) & (sub[:, 1] < y_hi) &
                (sub[:, 2] >= z_lo) & (sub[:, 2] < z_hi)
            )
            child_idx = indices[mask]
            if len(child_idx) == 0:
                children.append(None)
            else:
                children.append(_build(child_idx, depth + 1))

        return OctreeNode(center=center, half_diag=half_diag,
                          sample=sample, children=children, is_leaf=False)

    root = _build(np.arange(len(pts), dtype=np.intp), depth=0)
    if progress_cb:
        progress_cb("Octree built.")
    return root


def get_frustum_planes(renderer) -> np.ndarray:
    """Extract 6 clip planes from a VTK renderer using Gribb-Hartmann method.

    Returns (6, 4) float64 array; plane equation ax+by+cz+d — inside if >= 0.
    """
    camera = renderer.GetActiveCamera()
    aspect = renderer.GetTiledAspectRatio()
    mvp_vtk = camera.GetCompositeProjectionTransformMatrix(aspect, -1.0, 1.0)
    M = np.array(
        [[mvp_vtk.GetElement(r, c) for c in range(4)] for r in range(4)],
        dtype=np.float64,
    )

    planes = np.zeros((6, 4), dtype=np.float64)
    planes[0] = M[3] + M[0]   # left
    planes[1] = M[3] - M[0]   # right
    planes[2] = M[3] + M[1]   # bottom
    planes[3] = M[3] - M[1]   # top
    planes[4] = M[3] + M[2]   # near
    planes[5] = M[3] - M[2]   # far

    norms = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    planes /= norms
    return planes


def collect_indices(
    root: OctreeNode,
    camera_pos: np.ndarray,
    frustum_planes: np.ndarray,
    viewport_h: float,
    fov_half_tan: float,
    threshold_px: float = 80.0,
    max_pts: int = 4_000_000,
) -> np.ndarray:
    """DFS over the octree, frustum-culling and screen-size testing each node.

    Returns concatenated index array (into original pts).
    """
    collected: list[np.ndarray] = []
    total = [0]

    def _visit(node: OctreeNode) -> None:
        if total[0] >= max_pts:
            return

        c  = node.center
        hd = node.half_diag

        # Frustum cull: sphere vs each plane; d < -hd → entirely outside
        for plane in frustum_planes:
            d = plane[0]*c[0] + plane[1]*c[1] + plane[2]*c[2] + plane[3]
            if d < -hd:
                return

        # Screen-size test
        dist = float(np.linalg.norm(camera_pos - c))
        if dist < 1e-6:
            dist = 1e-6
        ss = hd / dist * viewport_h / (2.0 * max(fov_half_tan, 1e-6))

        if ss < threshold_px or node.is_leaf:
            rem  = max_pts - total[0]
            take = node.sample[:rem]
            collected.append(take)
            total[0] += len(take)
        else:
            for child in node.children:
                if child is not None:
                    _visit(child)

    _visit(root)

    if not collected:
        return np.empty(0, dtype=np.intp)
    return np.concatenate(collected).astype(np.intp)


def adaptive_point_size(n: int) -> float:
    """Return a point-size hint based on how many points are on screen."""
    if   n <    50_000: return 5.0
    elif n <   200_000: return 4.0
    elif n <   600_000: return 3.0
    elif n < 2_000_000: return 2.0
    else:               return 1.5
