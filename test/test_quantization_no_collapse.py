#!/usr/bin/env python3
import numpy as np
from voxel_viewer.file_compare import round_points_to_voxel, compute_two_file_diff


def test_centers_do_not_collapse_with_floor_quantization():
    v = 1.0
    origin = np.array([0.0, 0.0, 0.0])
    # Two adjacent voxel centers
    centers = np.array([
        origin + np.array([0.5, 0.5, 0.5]) * v,
        origin + np.array([1.5, 0.5, 0.5]) * v,
    ])
    q = round_points_to_voxel(centers, v, origin)
    # Expect distinct cell corners: [0,0,0] and [1,0,0]
    assert np.unique(q, axis=0).shape[0] == 2


def test_two_file_diff_keeps_both_centers():
    v = 1.0
    origin = np.array([0.0, 0.0, 0.0])
    file_pts = np.array([
        origin + np.array([0.5, 0.5, 0.5]) * v,
        origin + np.array([1.5, 0.5, 0.5]) * v,
    ])
    raw_pts = file_pts.copy()
    pts, cols = compute_two_file_diff(file_pts, raw_pts, v, origin=origin)
    # Common points should be 2 after quantization
    whites = (cols == np.array([1.0, 1.0, 1.0])).all(axis=1).sum()
    assert whites == 2 and pts.shape[0] == 2

