"""
Level-of-Detail (LOD) management for large point clouds.

Builds a 5-level resolution pyramid at load time (runs in a QThread).
At render time, selects the appropriate level based on camera distance,
so the GPU always receives ≈ 2-3 M points regardless of file size.

Level 0 (finest)  — full density   up to ~8 M pts — zoomed right in
Level 1           — 15 mm voxels   up to  3 M pts
Level 2           — 50 mm voxels   up to  1 M pts
Level 3           — 150 mm voxels  up to 300 K pts
Level 4 (coarsest)— 500 mm voxels  up to  80 K pts — full overview
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

# (voxel_size_m, max_points)  — ordered finest → coarsest
_LEVEL_CONFIGS: list[tuple[float, Optional[int]]] = [
    (0.005, 8_000_000),
    (0.015, 3_000_000),
    (0.050, 1_000_000),
    (0.150,   300_000),
    (0.500,    80_000),
]

# Camera distance / cloud radius → LOD level (finest = 0)
_DISTANCE_BREAKS = [0.4, 1.2, 3.5, 9.0]   # ratios; beyond last → level 4


@dataclass
class LODLevel:
    voxel_m:   float
    indices:   np.ndarray    # int64 indices into the full pts array
    max_pts:   Optional[int]

    @property
    def n(self) -> int:
        return len(self.indices)


@dataclass
class PointCloudLOD:
    pts:     np.ndarray              # (N, 3) original float64 points — read-only
    center:  np.ndarray = field(init=False)
    radius:  float       = field(init=False)
    levels:  list[LODLevel] = field(default_factory=list)

    def __post_init__(self):
        self.center = self.pts.mean(axis=0)
        deltas      = self.pts - self.center
        self.radius = float(np.sqrt((deltas ** 2).sum(axis=1)).max())

    def build(self, progress_cb: Callable[[str], None] | None = None) -> None:
        """Build all LOD levels.  Call this from a background thread."""
        self.levels.clear()
        for i, (voxel_m, max_pts) in enumerate(_LEVEL_CONFIGS):
            if progress_cb:
                progress_cb(
                    f"Building LOD {i + 1}/{len(_LEVEL_CONFIGS)}: "
                    f"{voxel_m * 1000:.0f} mm voxels…"
                )
            idx = _voxel_subsample(self.pts, voxel_m)
            if max_pts and len(idx) > max_pts:
                rng = np.random.default_rng(seed=i)
                idx = rng.choice(idx, max_pts, replace=False)
            self.levels.append(LODLevel(
                voxel_m=voxel_m,
                indices=np.sort(idx).astype(np.intp),
                max_pts=max_pts,
            ))

    def select(
        self,
        camera_pos:  np.ndarray,
        clip_mask:   Optional[np.ndarray] = None,
        force_level: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Return (indices_into_pts, lod_level_index) for the current camera.

        Parameters
        ----------
        camera_pos  : world-space camera position (3,)
        clip_mask   : bool (N,) — True = visible.  Applied after LOD selection.
        force_level : override automatic level selection (0=finest, 4=coarsest)
        """
        if not self.levels:
            idx = np.arange(len(self.pts), dtype=np.intp)
            return (idx if clip_mask is None else idx[clip_mask]), 0

        if force_level is not None:
            lv = max(0, min(force_level, len(self.levels) - 1))
        else:
            dist  = float(np.linalg.norm(camera_pos - self.center))
            ratio = dist / max(self.radius, 1e-3)
            lv    = len(_DISTANCE_BREAKS)          # default: coarsest
            for i, t in enumerate(_DISTANCE_BREAKS):
                if ratio <= t:
                    lv = i
                    break
            lv = min(lv, len(self.levels) - 1)

        idx = self.levels[lv].indices

        if clip_mask is not None:
            # Filter to only visible (clipped) points
            idx = idx[clip_mask[idx]]

        return idx, lv


# ── Fast numpy voxel subsampling ──────────────────────────────────────────────

def _voxel_subsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Return one representative point index per occupied voxel.

    Implementation: quantise XYZ to voxel grid, then np.unique on a
    void-view of the int32 triplets — O(N log N), no external deps.
    """
    q = np.floor(pts / voxel_size).astype(np.int32)
    # View each (i, j, k) row as 12 raw bytes so np.unique compares all 3 axes
    dt   = np.dtype((np.void, q.dtype.itemsize * 3))
    flat = np.ascontiguousarray(q).view(dt).ravel()
    _, first_idx = np.unique(flat, return_index=True)
    return first_idx


def quick_sample(pts: np.ndarray, target: int = 200_000) -> np.ndarray:
    """Return a uniform-stride subsample for instant preview before LOD builds."""
    n    = len(pts)
    step = max(1, n // target)
    return np.arange(0, n, step, dtype=np.intp)


def adaptive_point_size(n_rendered: int) -> float:
    """Slightly enlarge points when density is low so they tile the screen."""
    if   n_rendered <    50_000: return 5.0
    elif n_rendered <   200_000: return 4.0
    elif n_rendered <   600_000: return 3.0
    elif n_rendered < 2_000_000: return 2.0
    else:                        return 1.5
