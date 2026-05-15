import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from processing.geometry import fit_cylinder_ransac, fit_plane_ransac


def make_cylinder(radius=10.0, height=20.0, n=2000, noise=0.005):
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, n)
    z     = rng.uniform(0, height, n)
    x = radius * np.cos(theta) + rng.normal(0, noise, n)
    y = radius * np.sin(theta) + rng.normal(0, noise, n)
    return np.column_stack([x, y, z])


def make_plane(z=0.0, extent=10.0, n=1000, noise=0.005):
    rng = np.random.default_rng(1)
    x  = rng.uniform(-extent, extent, n)
    y  = rng.uniform(-extent, extent, n)
    zs = np.full(n, z) + rng.normal(0, noise, n)
    return np.column_stack([x, y, zs])


def test_cylinder_radius():
    model = fit_cylinder_ransac(make_cylinder(radius=10.0), radial_tolerance=0.03)
    assert model is not None
    assert abs(model.radius - 10.0) < 0.2


def test_cylinder_center():
    pts = make_cylinder(radius=5.0) + [3.0, -2.0, 0.0]
    model = fit_cylinder_ransac(pts, radial_tolerance=0.03)
    assert model is not None
    assert abs(model.axis_point[0] - 3.0) < 0.3
    assert abs(model.axis_point[1] + 2.0) < 0.3


def test_plane_normal_vertical():
    model = fit_plane_ransac(make_plane(z=5.0))
    assert model is not None
    assert abs(abs(model.normal[2]) - 1.0) < 0.05


def test_plane_inlier_fraction():
    model = fit_plane_ransac(make_plane(z=0.0, n=1000))
    assert model is not None
    assert model.inlier_mask.sum() / 1000 > 0.85
