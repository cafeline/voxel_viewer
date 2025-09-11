#!/usr/bin/env python3
import numpy as np
from voxel_viewer.mesh_utils import build_cubes_as_arrays


def test_build_cubes_as_arrays_shapes_and_bounds():
    centers = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    size = 1.0
    verts, tris = build_cubes_as_arrays(centers, size)
    # Each cube: 8 verts, 12 tris
    assert verts.shape == (16, 3)
    assert tris.shape == (24, 3)
    # Bounds: first cube spans [-0.5, +0.5], second spans [1.5, 2.5]
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    np.testing.assert_allclose(vmin, [-0.5, -0.5, -0.5])
    np.testing.assert_allclose(vmax, [2.5, 0.5, 0.5])

