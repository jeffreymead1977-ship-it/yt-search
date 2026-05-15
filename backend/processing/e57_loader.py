"""Read an E57 file and return a single merged (N,3) XYZ numpy array."""
import pye57
import numpy as np
from pathlib import Path


def load_e57(path: str | Path) -> tuple[np.ndarray, dict]:
    path = Path(path)
    e57 = pye57.E57(str(path))
    header = e57.get_header(0)

    metadata = {
        "file": path.name,
        "scan_count": e57.scan_count,
        "guid": str(getattr(header, "guid", "")),
    }

    clouds = []
    for i in range(e57.scan_count):
        data = e57.read_scan(i, ignore_missing_fields=True)
        xyz = np.column_stack([
            data["cartesianX"],
            data["cartesianY"],
            data["cartesianZ"],
        ]).astype(np.float64)
        # Drop invalid points (NaN or cartesianInvalidState != 0)
        if "cartesianInvalidState" in data:
            valid = data["cartesianInvalidState"] == 0
            xyz = xyz[valid]
        finite = np.all(np.isfinite(xyz), axis=1)
        clouds.append(xyz[finite])

    return np.concatenate(clouds, axis=0), metadata
