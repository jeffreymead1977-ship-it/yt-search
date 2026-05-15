"""Verify RANSAC cylinder and plane fitting on synthetic data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from processing.geometry import fit_cylinder_ransac, fit_plane_ransac


def make_cylinder(radius=10.0, height=20.0, n=2000, noise=0.005):
    """Generate points on a vertical cylinder surface."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(0, height, n)
    x = radius * np.cos(theta) + rng.normal(0, noise, n)
    y = radius * np.sin(theta) + rng.normal(0, noise, n)
    return np.column_stack([x, y, z])


def make_plane(z=0.0, extent=10.0, n=1000, noise=0.005):
    """Generate points on a horizontal plane."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-extent, extent, n)
    y = rng.uniform(-extent, extent, n)
    z_pts = np.full(n, z) + rng.normal(0, noise, n)
    return np.column_stack([x, y, z_pts])


def test_cylinder_radius():
    pts = make_cylinder(radius=10.0)
    model = fit_cylinder_ransac(pts, radial_tolerance=0.03)
    assert model is not None, "Cylinder RANSAC returned None"
    assert abs(model.radius - 10.0) < 0.2, f"Radius off: {model.radius:.3f}"


def test_cylinder_center():
    pts = make_cylinder(radius=5.0) + np.array([3.0, -2.0, 0.0])
    model = fit_cylinder_ransac(pts, radial_tolerance=0.03)
    assert model is not None
    cx, cy = model.axis_point[0], model.axis_point[1]
    assert abs(cx - 3.0) < 0.3, f"cx off: {cx:.3f}"
    assert abs(cy + 2.0) < 0.3, f"cy off: {cy:.3f}"


def test_plane_normal_vertical():
    pts = make_plane(z=5.0)
    model = fit_plane_ransac(pts)
    assert model is not None
    assert abs(abs(model.normal[2]) - 1.0) < 0.05, f"Normal not vertical: {model.normal}"


def test_plane_inlier_fraction():
    pts = make_plane(z=0.0, n=1000)
    model = fit_plane_ransac(pts)
    assert model is not None
    frac = model.inlier_mask.sum() / len(pts)
    assert frac > 0.85, f"Too few inliers: {frac:.2f}"
