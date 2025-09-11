#!/usr/bin/env python3
"""Color utilities for height-aware shading of common (white) voxels."""

import numpy as np
from typing import List, Tuple


def _percentile_range(z: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> Tuple[float, float]:
    if z.size == 0:
        return 0.0, 1.0
    zl = float(np.percentile(z, lo))
    zh = float(np.percentile(z, hi))
    if zh <= zl:
        zh = zl + 1.0
    return zl, zh


def shade_whites_by_height(
    white_points: np.ndarray,
    bands: int = 8,
    min_brightness: float = 0.85,
    max_brightness: float = 1.00,
) -> List[Tuple[np.ndarray, list]]:
    """Split white points into banded groups by Z and assign subtle brightness.

    Returns a list of (centers ndarray Nx3, color [r,g,b]) suitable for update_cubes.
    If there are no white points, returns an empty list.
    """
    if white_points is None or len(white_points) == 0:
        return []
    wp = np.asarray(white_points)
    z = wp[:, 2]
    zl, zh = _percentile_range(z)
    # Normalize and quantize to bands
    t = np.clip((z - zl) / (zh - zl), 0.0, 1.0)
    # Avoid too many tiny groups: clamp bands between 2 and 16
    b = max(2, min(int(bands), 16))
    q = np.floor(t * b).astype(int)
    q[q == b] = b - 1

    # Precompute brightness for each band
    # Option A: linear ramp; could swap to gamma if desired
    br = np.linspace(min_brightness, max_brightness, b)

    out: List[Tuple[np.ndarray, list]] = []
    for band in range(b):
        mask = (q == band)
        if not np.any(mask):
            continue
        pts = wp[mask]
        f = float(br[band])
        col = [f, f, f]
        out.append((pts, col))
    return out


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """Convert HSV (0-1) to RGB (0-1)."""
    if s <= 1e-12:
        return v, v, v
    h = (h % 1.0) * 6.0
    i = int(np.floor(h))
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return float(r), float(g), float(b)


def shade_whites_vivid_by_height(
    white_points: np.ndarray,
    bands: int = 64,
    sat: float = 0.95,
    val: float = 1.0,
) -> List[Tuple[np.ndarray, list]]:
    """Split white points by height and assign vivid rainbow colors.

    - Hue traverses 0..1 across height (with small offset for balance)
    - Saturation/Value are kept high for鮮やかさ
    """
    if white_points is None or len(white_points) == 0:
        return []
    wp = np.asarray(white_points)
    z = wp[:, 2]
    zl, zh = _percentile_range(z)
    t = np.clip((z - zl) / (zh - zl), 0.0, 1.0)
    # Use many small bands to approximate continuous gradient
    b = max(16, min(int(bands), 256))
    q = np.floor(t * b).astype(int)
    q[q == b] = b - 1

    # color per band: hue across almost full rainbow [0, 0.92]
    hues = np.linspace(0.0, 0.92, b, endpoint=True)
    cols = [_hsv_to_rgb(h, sat, val) for h in hues]

    out: List[Tuple[np.ndarray, list]] = []
    for band in range(b):
        mask = (q == band)
        if not np.any(mask):
            continue
        pts = wp[mask]
        col = list(cols[band])
        out.append((pts, col))
    return out
