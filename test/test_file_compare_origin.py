#!/usr/bin/env python3
import numpy as np
from voxel_viewer.file_compare import round_points_to_voxel, compute_two_file_diff


def test_round_points_to_voxel_with_origin():
    # Voxel size and non-zero origin
    v = 2.0
    origin = np.array([10.0, -5.0, 1.0])
    # Points near the origin-offset voxel centers
    pts = np.array([
        origin + np.array([0.49, 0.49, 0.49]) * v,   # -> origin + 0*v (floor)
        origin + np.array([1.51, -0.49, 2.51]) * v   # -> origin + [2, -2, 4] (floor)
    ])
    out = round_points_to_voxel(pts, v, origin)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0], origin + np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(out[1], origin + np.array([2.0, -2.0, 4.0]))


def test_compute_two_file_diff_with_origin():
    v = 1.0
    origin = np.array([100.0, 50.0, -20.0])
    # Define three points: A common, B only in file, C only in raw
    A = origin + np.array([0.1, 0.2, 0.4])
    B = origin + np.array([1.2, 0.1, 0.4])
    C = origin + np.array([0.1, 1.1, 0.4])
    file_pts = np.vstack([A, B])
    raw_pts = np.vstack([A, C])
    pts, cols = compute_two_file_diff(file_pts, raw_pts, v, origin=origin)
    # Expect 3 outputs, with one of each color
    assert pts.shape[0] == 3
    whites = (cols == np.array([1.0, 1.0, 1.0])).all(axis=1).sum()
    reds = (cols == np.array([1.0, 0.0, 0.0])).all(axis=1).sum()
    greens = (cols == np.array([0.0, 1.0, 0.0])).all(axis=1).sum()
    assert whites == 1 and reds == 1 and greens == 1
