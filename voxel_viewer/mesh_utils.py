#!/usr/bin/env python3
"""Helpers for building cube meshes efficiently."""

import numpy as np


def build_cubes_as_arrays(centers: np.ndarray, size: float) -> tuple:
    """Return (vertices Nx3, triangles Mx3) for axis-aligned cubes.

    - centers: ndarray (N,3) of cube centers
    - size: edge length (float)

    The cubes do not share vertices (simple instancing-style replication):
    - each cube contributes 8 vertices and 12 triangles
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

    N = centers.shape[0]
    h = float(size) / 2.0

    # 8 corner offsets for a unit cube centered at origin
    offsets = np.array([
        [-h, -h, -h],
        [-h, -h,  h],
        [-h,  h, -h],
        [-h,  h,  h],
        [ h, -h, -h],
        [ h, -h,  h],
        [ h,  h, -h],
        [ h,  h,  h],
    ], dtype=np.float64)

    # base triangles (12) using the 8 vertices
    base_tris = np.array([
        [0, 2, 1], [1, 2, 3],  # -X face
        [4, 5, 6], [5, 7, 6],  # +X face
        [0, 1, 4], [1, 5, 4],  # -Y face
        [2, 6, 3], [3, 6, 7],  # +Y face
        [0, 4, 2], [2, 4, 6],  # -Z face
        [1, 3, 5], [3, 7, 5],  # +Z face
    ], dtype=np.int32)

    # Repeat offsets for all centers and add
    verts = (offsets.reshape(1, 8, 3) + centers.reshape(N, 1, 3)).reshape(N * 8, 3)

    # Triangles indices with offset per cube
    tri_offsets = (np.arange(N, dtype=np.int32) * 8).reshape(N, 1, 1)
    tris = (base_tris.reshape(1, 12, 3) + tri_offsets).reshape(N * 12, 3)

    return verts, tris

