#!/usr/bin/env python3
import numpy as np
from voxel_viewer.file_compare import compute_two_file_diff, round_points_to_voxel


def test_round_points_to_voxel():
    pts = np.array([[0.49, 0.51, 1.49], [1.01, -0.49, 2.51]])
    out = round_points_to_voxel(pts, 1.0)
    assert out.shape == (2, 3)
    assert np.allclose(out[0], [0.0, 1.0, 1.0])
    assert np.allclose(out[1], [1.0, -0.0, 3.0])


def test_compute_two_file_diff():
    # file: A, B; raw: A, C (voxel_size=1)
    A = np.array([[0.1, 0.2, 0.4]])
    B = np.array([[1.2, 0.1, 0.4]])
    C = np.array([[0.1, 1.1, 0.4]])
    file_pts = np.vstack([A, B])
    raw_pts = np.vstack([A, C])
    pts, cols = compute_two_file_diff(file_pts, raw_pts, 1.0)
    # Expect 3 points total: A (white), B (red), C (green)
    assert pts.shape[0] == 3
    # Count colors
    whites = (cols == np.array([1, 1, 1])).all(axis=1).sum()
    reds = (cols == np.array([1, 0, 0])).all(axis=1).sum()
    greens = (cols == np.array([0, 1, 0])).all(axis=1).sum()
    assert whites == 1 and reds == 1 and greens == 1

