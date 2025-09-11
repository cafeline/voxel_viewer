#!/usr/bin/env python3
import numpy as np
from voxel_viewer.color_utils import shade_whites_by_height
from voxel_viewer.color_utils import shade_whites_vivid_by_height


def test_shade_whites_by_height_bands_and_bounds():
    # build points with increasing z
    z = np.linspace(0.0, 10.0, 100)
    pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    sets = shade_whites_by_height(pts, bands=8, min_brightness=0.85, max_brightness=1.0)
    # should have at least 2 groups (often 8, but tolerate empty bands)
    assert len(sets) >= 2
    # colors are grayscale within [0.85,1.0]
    for _, c in sets:
        assert 0.85 <= c[0] <= 1.0 and c[0] == c[1] == c[2]
    # concatenated points should reconstruct original size
    total = sum(s[0].shape[0] for s in sets)
    assert total == pts.shape[0]


def test_shade_whites_vivid_by_height_colors_are_not_gray():
    z = np.linspace(0.0, 10.0, 100)
    pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    sets = shade_whites_vivid_by_height(pts, bands=10)
    # enough bands to be colorful
    assert len(sets) >= 5
    # colors should not be grayscale; check at least one has r!=g or g!=b
    distinct_colors = 0
    for _, c in sets:
        if not (abs(c[0]-c[1]) < 1e-6 and abs(c[1]-c[2]) < 1e-6):
            distinct_colors += 1
    assert distinct_colors >= 3
