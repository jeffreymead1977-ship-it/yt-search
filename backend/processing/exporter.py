"""Export segmentation results to labeled PLY and JSON summary."""
import json
import numpy as np
from pathlib import Path
from models.tank_scan import SegmentationResult, Label, LABEL_COLORS


def export_labeled_ply(result: SegmentationResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    pts = result.points
    labels = result.labels
    n = len(pts)

    colors = np.zeros((n, 3), dtype=np.float32)
    for label, color in LABEL_COLORS.items():
        mask = labels == int(label)
        colors[mask] = color

    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar label\n")
        f.write("end_header\n")
        rgb = (colors * 255).astype(np.uint8)
        for i in range(n):
            x, y, z = pts[i]
            r, g, b = rgb[i]
            lbl = int(labels[i])
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b} {lbl}\n")

    return output_path


def export_summary(result: SegmentationResult, output_path: str | Path) -> dict:
    output_path = Path(output_path)
    cyl = result.cylinder

    summary = {
        "point_counts": result.point_counts(),
        "total_points": int(len(result.points)),
        "cylinder_radius_m": round(cyl.radius_m, 4) if cyl else None,
        "cylinder_height_m": round(cyl.height_m, 4) if cyl else None,
        "floor_elevation_m": round(result.floor_z_m, 4) if result.floor_z_m is not None else None,
        "roof_elevation_m": round(result.roof_z_m, 4) if result.roof_z_m is not None else None,
        "scan_metadata": result.scan_metadata,
        "label_legend": {label.name: int(label) for label in Label},
    }

    output_path.write_text(json.dumps(summary, indent=2))
    return summary
