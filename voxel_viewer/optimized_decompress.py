#!/usr/bin/env python3
"""Vectorized helpers to decompress dictionary-encoded voxel blocks.

This module provides fast paths for converting dictionary patterns to
relative coordinates and reconstructing world-space centers for many blocks
at once using NumPy broadcasting and preallocation.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple
import multiprocessing as mp


def unpack_pattern_coords(pattern_bytes: np.ndarray, pattern_length: int, block_size: int) -> np.ndarray:
    """Return relative voxel coordinates (K,3) for set bits in the pattern.

    - pattern_bytes: 1D uint8 array containing the pattern's bytes
    - pattern_length: number of meaningful bits in the pattern
    - block_size: edge length of the block (voxels)
    """
    # Bits: little-endian to match encoder
    bits = np.unpackbits(np.asarray(pattern_bytes, dtype=np.uint8), bitorder='little')[:pattern_length]
    idx = np.nonzero(bits)[0]
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.int16)
    bs = int(block_size)
    b2 = bs * bs
    z = idx // b2
    y = (idx % b2) // bs
    x = idx % bs
    return np.stack((x, y, z), axis=1).astype(np.int16)


def vectorized_points_from_blocks(
    voxel_positions: np.ndarray,
    indices: np.ndarray,
    dictionary_patterns: np.ndarray,
    pattern_length: int,
    block_size: int,
    grid_origin: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Reconstruct world-space voxel centers for blocks using vectorization.

    - voxel_positions: (M,3) int32 block indices
    - indices: (M,) int32 pattern ids per block
    - dictionary_patterns: flat uint8 array of concatenated patterns
    - pattern_length: bits per pattern
    - block_size: edge length (voxels)
    - grid_origin: (3,) float64 origin
    - voxel_size: float
    Returns: (N,3) float64 centers
    """
    if voxel_positions is None or len(voxel_positions) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    M = int(voxel_positions.shape[0])
    bs = int(block_size)
    bytes_per_pattern = (int(pattern_length) + 7) // 8

    # Unique pattern ids present
    inds = np.asarray(indices, dtype=np.int32).reshape(-1)
    unique_pids, inv = np.unique(inds, return_inverse=False), None

    # Precompute pattern -> coords and popcounts
    coords_map: Dict[int, np.ndarray] = {}
    pop_map: Dict[int, int] = {}
    for pid in unique_pids:
        start = int(pid) * bytes_per_pattern
        end = start + bytes_per_pattern
        pbytes = dictionary_patterns[start:end]
        coords = unpack_pattern_coords(pbytes, int(pattern_length), bs)
        coords_map[int(pid)] = coords
        pop_map[int(pid)] = int(coords.shape[0])

    # Total points to preallocate
    total = 0
    for pid in unique_pids:
        count_blocks = int((inds == pid).sum())
        total += count_blocks * pop_map[int(pid)]
    if total == 0:
        return np.zeros((0, 3), dtype=np.float64)
    out = np.empty((total, 3), dtype=np.float64)

    # Compute block origins once
    voxel_positions = np.asarray(voxel_positions, dtype=np.int32)
    grid_origin = np.asarray(grid_origin, dtype=np.float64).reshape(3)
    block_origins_all = grid_origin + (voxel_positions.astype(np.float64) * (bs * float(voxel_size)))

    # Fill per pattern group
    offset = 0
    for pid in unique_pids:
        mask = (inds == pid)
        if not mask.any():
            continue
        origins = block_origins_all[mask]  # (B,3)
        coords = coords_map[int(pid)]      # (K,3)
        if coords.size == 0:
            continue
        # Broadcast add: (B,1,3)+(1,K,3)
        pts = origins[:, None, :] + (coords.astype(np.float64)[None, :, :] + 0.5) * float(voxel_size)
        B, K = origins.shape[0], coords.shape[0]
        n = B * K
        out[offset:offset + n] = pts.reshape(n, 3)
        offset += n

    if offset != total:
        out = out[:offset]
    return out


def _compute_pts_for_pid(args: Tuple[int, np.ndarray, np.ndarray, float, np.ndarray]) -> np.ndarray:
    pid, origins, coords, voxel_size, grid_origin = args
    if coords.size == 0 or origins.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts = origins[:, None, :] + (coords[None, :, :].astype(np.float32) + 0.5) * float(voxel_size)
    return pts.reshape(-1, 3).astype(np.float64, copy=False)


def vectorized_points_from_blocks_parallel(
    voxel_positions: np.ndarray,
    indices: np.ndarray,
    dictionary_patterns: np.ndarray,
    pattern_length: int,
    block_size: int,
    grid_origin: np.ndarray,
    voxel_size: float,
    processes: int | None = None,
) -> np.ndarray:
    """Parallel version across pattern ids (multiprocessing).

    Note: For small data or few patterns, serial may be faster. Use when unique
    pattern ids or blocks are large.
    """
    if voxel_positions is None or len(voxel_positions) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    bs = int(block_size)
    bytes_per_pattern = (int(pattern_length) + 7) // 8
    inds = np.asarray(indices, dtype=np.int32).reshape(-1)
    unique_pids = np.unique(inds)

    # Precompute coords per pattern id (once in parent)
    coords_map: Dict[int, np.ndarray] = {}
    for pid in unique_pids:
        start = int(pid) * bytes_per_pattern
        end = start + bytes_per_pattern
        pbytes = dictionary_patterns[start:end]
        coords_map[int(pid)] = unpack_pattern_coords(pbytes, int(pattern_length), bs).astype(np.int16, copy=False)

    grid_origin = np.asarray(grid_origin, dtype=np.float64).reshape(3)
    block_origins_all = grid_origin + (np.asarray(voxel_positions, dtype=np.float32) * (bs * float(voxel_size)))

    tasks = []
    for pid in unique_pids:
        mask = (inds == pid)
        if mask.any():
            origins = block_origins_all[mask]
            coords = coords_map[int(pid)].astype(np.float32, copy=False)
            tasks.append((int(pid), origins, coords, float(voxel_size), grid_origin))

    if not tasks:
        return np.zeros((0, 3), dtype=np.float64)
    with mp.Pool(processes=processes) as pool:
        parts = pool.map(_compute_pts_for_pid, tasks)
    if not parts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(parts)
