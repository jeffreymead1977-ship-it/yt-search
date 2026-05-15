"""Export labeled PLY (binary) and JSON summary."""
import json
import numpy as np
from pathlib import Path
from models.tank_scan import SegmentationResult, Label, LABEL_COLORS


def export_labeled_ply(result: SegmentationResult, output_path: str | Path) -> Path:
    """
    Write a binary little-endian PLY with per-point XYZ, RGB and label.

    Binary PLY is ~100x faster to write than ASCII and ~10x smaller on disk.
    CloudCompare, MeshLab, and most inspection tools read it without issues.
    """
    output_path = Path(output_path)
    pts    = result.points.astype(np.float32)
    labels = result.labels
    n      = len(pts)

    colors = np.zeros((n, 3), dtype=np.uint8)
    for label, rgb in LABEL_COLORS.items():
        colors[labels == int(label)] = rgb

    # Pack into a structured array so the whole block writes in one call
    dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('label', 'u1'),
    ])
    data = np.empty(n, dtype=dtype)
    data['x'] = pts[:, 0];  data['y'] = pts[:, 1];  data['z'] = pts[:, 2]
    data['red']   = colors[:, 0]
    data['green'] = colors[:, 1]
    data['blue']  = colors[:, 2]
    data['label'] = labels.astype(np.uint8)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar label\n"
        "end_header\n"
    )

    with open(output_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())

    return output_path


def export_summary(result: SegmentationResult, output_path: str | Path) -> dict:
    output_path = Path(output_path)
    cyl = result.cylinder
    summary = {
        "scan_type":         result.scan_type,
        "total_points":      int(len(result.points)),
        "point_counts":      result.point_counts(),
        "cylinder_radius_m": round(cyl.radius_m, 4) if cyl else None,
        "cylinder_height_m": round(cyl.height_m, 4) if cyl else None,
        "floor_elevation_m": round(result.floor_z_m, 4) if result.floor_z_m is not None else None,
        "roof_elevation_m":  round(result.roof_z_m,  4) if result.roof_z_m  is not None else None,
        "scan_metadata":     result.scan_metadata,
        "label_legend":      {label.name: int(label) for label in Label},
    }
    output_path.write_text(json.dumps(summary, indent=2))
    return summary
