"""Export labeled PLY and JSON summary."""
import json
import numpy as np
from pathlib import Path
from models.tank_scan import SegmentationResult, Label, LABEL_COLORS


def export_labeled_ply(result: SegmentationResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    pts    = result.points
    labels = result.labels
    n      = len(pts)

    colors = np.zeros((n, 3), dtype=np.uint8)
    for label, rgb in LABEL_COLORS.items():
        colors[labels == int(label)] = rgb

    with open(output_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        for prop in ("x", "y", "z"):
            f.write(f"property float {prop}\n")
        for prop in ("red", "green", "blue"):
            f.write(f"property uchar {prop}\n")
        f.write("property uchar label\nend_header\n")
        for i in range(n):
            x, y, z = pts[i]
            r, g, b = colors[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b} {int(labels[i])}\n")

    return output_path


def export_summary(result: SegmentationResult, output_path: str | Path) -> dict:
    output_path = Path(output_path)
    cyl = result.cylinder
    summary = {
        "scan_type":        result.scan_type,
        "total_points":     int(len(result.points)),
        "point_counts":     result.point_counts(),
        "cylinder_radius_m": round(cyl.radius_m, 4) if cyl else None,
        "cylinder_height_m": round(cyl.height_m, 4) if cyl else None,
        "floor_elevation_m": round(result.floor_z_m, 4) if result.floor_z_m is not None else None,
        "roof_elevation_m":  round(result.roof_z_m,  4) if result.roof_z_m  is not None else None,
        "scan_metadata":    result.scan_metadata,
        "label_legend":     {label.name: int(label) for label in Label},
    }
    output_path.write_text(json.dumps(summary, indent=2))
    return summary
