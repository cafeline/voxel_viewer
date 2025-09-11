#!/usr/bin/env python3
"""Pure helper for file-to-file voxel comparison (testable without ROS/Open3D)."""

import numpy as np
from typing import Tuple


def round_points_to_voxel(points: np.ndarray, voxel_size: float, origin: np.ndarray = None) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.zeros((0, 3))
    if voxel_size <= 0:
        return np.asarray(points, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    if origin is None:
        # Use floor quantization to the voxel cell, avoiding banker's rounding collapse
        return np.floor(p / voxel_size) * voxel_size
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    # Quantize to the lower cell corner relative to origin
    return np.floor((p - o) / voxel_size) * voxel_size + o


def compute_two_file_diff(
    file_pts: np.ndarray,
    raw_pts: np.ndarray,
    voxel_size: float,
    origin: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute colored points for file-to-file comparison.

    Returns a tuple (points Nx3, colors Nx3) where colors are:
    - common: white [1,1,1]
    - file-only: red [1,0,0]
    - raw-only: green [0,1,0]
    """
    f1 = round_points_to_voxel(file_pts, voxel_size, origin)
    f2 = round_points_to_voxel(raw_pts, voxel_size, origin)
    set1 = set(map(tuple, f1))
    set2 = set(map(tuple, f2))
    common = set1 & set2
    only_f1 = set1 - set2
    only_f2 = set2 - set1

    pts = []
    cols = []
    for p in common:
        pts.append(p)
        cols.append([1.0, 1.0, 1.0])
    for p in only_f1:
        pts.append(p)
        cols.append([1.0, 0.0, 0.0])
    for p in only_f2:
        pts.append(p)
        cols.append([0.0, 1.0, 0.0])

    if pts:
        return np.array(pts, dtype=np.float64), np.array(cols, dtype=np.float64)
    return np.zeros((0, 3)), np.zeros((0, 3))


def validate_voxel_sizes(v1: float, v2: float, tol: float = 1e-9) -> bool:
    """Return True if two voxel sizes are equal within tolerance."""
    try:
        if v1 is None or v2 is None:
            return False
        return abs(float(v1) - float(v2)) <= tol
    except Exception:
        return False
