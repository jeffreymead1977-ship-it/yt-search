"""End-to-end segmenter test on a synthetic tank point cloud."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from models.tank_scan import Label
from processing.segmenter import segment


def make_tank(radius=8.0, height=15.0, n_shell=3000, n_floor=1000, n_roof=800, n_dead=200, n_noise=100):
    rng = np.random.default_rng(42)

    # Shell
    theta = rng.uniform(0, 2 * np.pi, n_shell)
    z = rng.uniform(0, height, n_shell)
    shell = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])
    shell += rng.normal(0, 0.005, shell.shape)

    # Floor
    r_f = rng.uniform(0, radius, n_floor)
    t_f = rng.uniform(0, 2 * np.pi, n_floor)
    floor = np.column_stack([r_f * np.cos(t_f), r_f * np.sin(t_f), np.zeros(n_floor)])
    floor += rng.normal(0, 0.005, floor.shape)

    # Roof
    r_r = rng.uniform(0, radius, n_roof)
    t_r = rng.uniform(0, 2 * np.pi, n_roof)
    roof = np.column_stack([r_r * np.cos(t_r), r_r * np.sin(t_r), np.full(n_roof, height)])
    roof += rng.normal(0, 0.005, roof.shape)

    # Deadwood (internal random points)
    r_d = rng.uniform(0, radius * 0.7, n_dead)
    t_d = rng.uniform(0, 2 * np.pi, n_dead)
    z_d = rng.uniform(0.5, height - 0.5, n_dead)
    dead = np.column_stack([r_d * np.cos(t_d), r_d * np.sin(t_d), z_d])

    # Noise (outside tank)
    noise = rng.uniform(-radius * 2, radius * 2, (n_noise, 3))
    noise[:, 2] = rng.uniform(-5, height + 5, n_noise)

    return np.concatenate([shell, floor, roof, dead, noise], axis=0)


def test_all_labels_assigned():
    xyz = make_tank()
    result = segment(xyz, voxel_size=0.05)
    counts = result.point_counts()
    for label_name in ["SHELL", "FLOOR", "ROOF"]:
        assert counts[label_name] > 0, f"{label_name} has zero points"


def test_cylinder_detected():
    xyz = make_tank(radius=8.0)
    result = segment(xyz, voxel_size=0.05)
    assert result.cylinder is not None, "No cylinder detected"
    assert abs(result.cylinder.radius_m - 8.0) < 1.5, f"Radius off: {result.cylinder.radius_m:.2f}"


def test_floor_elevation():
    xyz = make_tank(radius=8.0, height=15.0)
    result = segment(xyz, voxel_size=0.05)
    assert result.floor_z_m is not None
    assert abs(result.floor_z_m) < 0.5, f"Floor z off: {result.floor_z_m:.2f}"
