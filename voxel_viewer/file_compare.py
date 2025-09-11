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
    # Quantize to integer cell indices to avoid float set/tuple issues
    o = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64).reshape(3)
    def to_cells(pts):
        if pts is None or len(pts) == 0:
            return np.zeros((0, 3), dtype=np.int32)
        p = np.asarray(pts, dtype=np.float64)
        return np.floor((p - o) / voxel_size).astype(np.int32)

    c1 = to_cells(file_pts)
    c2 = to_cells(raw_pts)

    # Unique rows
    def unique_rows(a):
        if a.size == 0:
            return a
        a_view = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        _, idx = np.unique(a_view, return_index=True)
        return a[idx]

    u1 = unique_rows(c1)
    u2 = unique_rows(c2)

    # Set-like ops via concatenation + flags
    def diff_and_intersection(a, b):
        if a.size == 0 and b.size == 0:
            return (np.zeros((0,3), dtype=np.int32), np.zeros((0,3), dtype=np.int32), np.zeros((0,3), dtype=np.int32))
        if a.size == 0:
            return (np.zeros((0,3), dtype=np.int32), np.zeros((0,3), dtype=np.int32), b)
        if b.size == 0:
            return (a, np.zeros((0,3), dtype=np.int32), np.zeros((0,3), dtype=np.int32))
        ab = np.vstack([a, b])
        ab_view = np.ascontiguousarray(ab).view(np.dtype((np.void, ab.dtype.itemsize * ab.shape[1])))
        uniq, inv, counts = np.unique(ab_view, return_inverse=True, return_counts=True)
        # indices mapping: [0..len(a)-1] -> a, [len(a)..] -> b
        is_from_a = np.arange(len(ab)) < len(a)
        only_a_mask = (counts[inv] == 1) & is_from_a
        only_b_mask = (counts[inv] == 1) & (~is_from_a)
        common_mask = (counts[inv] > 1)
        return ab[only_a_mask], ab[only_b_mask], ab[common_mask]

    only1_cells, only2_cells, common_cells = diff_and_intersection(u1, u2)

    # Map back to world centers (lower corner + 0.5 voxel)
    def cells_to_centers(cells):
        if cells.size == 0:
            return np.zeros((0,3), dtype=np.float64)
        return o + (cells.astype(np.float64) + 0.5) * voxel_size

    only1 = cells_to_centers(only1_cells)
    only2 = cells_to_centers(only2_cells)
    common = cells_to_centers(common_cells)

    pts = []
    cols = []
    if common.size:
        pts.append(common); cols.append(np.tile([1.0,1.0,1.0], (len(common),1)))
    if only1.size:
        pts.append(only1); cols.append(np.tile([1.0,0.0,0.0], (len(only1),1)))
    if only2.size:
        pts.append(only2); cols.append(np.tile([0.0,1.0,0.0], (len(only2),1)))
    if pts:
        return np.vstack(pts), np.vstack(cols)
    return np.zeros((0,3)), np.zeros((0,3))


def validate_voxel_sizes(v1: float, v2: float, atol: float = 1e-7, rtol: float = 1e-6) -> bool:
    """Return True if two voxel sizes are equal within absolute/relative tolerance."""
    try:
        if v1 is None or v2 is None:
            return False
        a = float(v1); b = float(v2)
        return abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))
    except Exception:
        return False
