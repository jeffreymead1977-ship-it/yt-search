import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from models.tank_scan import Label
from processing.segmenter import segment


def make_tank(radius=8.0, height=15.0):
    rng = np.random.default_rng(42)

    theta = rng.uniform(0, 2 * np.pi, 3000)
    shell = np.column_stack([
        radius * np.cos(theta), radius * np.sin(theta), rng.uniform(0, height, 3000)
    ]) + rng.normal(0, 0.005, (3000, 3))

    r_f = rng.uniform(0, radius, 1000)
    t_f = rng.uniform(0, 2 * np.pi, 1000)
    floor = np.column_stack([r_f * np.cos(t_f), r_f * np.sin(t_f), np.zeros(1000)])
    floor += rng.normal(0, 0.005, floor.shape)

    r_r = rng.uniform(0, radius, 800)
    t_r = rng.uniform(0, 2 * np.pi, 800)
    roof = np.column_stack([r_r * np.cos(t_r), r_r * np.sin(t_r), np.full(800, height)])
    roof += rng.normal(0, 0.005, roof.shape)

    r_d = rng.uniform(0, radius * 0.7, 200)
    t_d = rng.uniform(0, 2 * np.pi, 200)
    dead = np.column_stack([r_d * np.cos(t_d), r_d * np.sin(t_d), rng.uniform(0.5, height - 0.5, 200)])

    noise = rng.uniform(-radius * 2, radius * 2, (100, 3))
    return np.concatenate([shell, floor, roof, dead, noise])


def test_all_main_labels_present():
    result = segment(make_tank(), voxel_size=0.05)
    counts = result.point_counts()
    for name in ["SHELL", "FLOOR", "ROOF"]:
        assert counts[name] > 0, f"{name} has zero points"


def test_cylinder_detected():
    result = segment(make_tank(radius=8.0), voxel_size=0.05)
    assert result.cylinder is not None
    assert abs(result.cylinder.radius_m - 8.0) < 1.5


def test_floor_near_zero():
    result = segment(make_tank(), voxel_size=0.05)
    assert result.floor_z_m is not None
    assert abs(result.floor_z_m) < 0.5
