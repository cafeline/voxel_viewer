#!/usr/bin/env python3
from voxel_viewer.file_compare import validate_voxel_sizes


def test_validate_voxel_sizes_with_atol_rtol():
    assert validate_voxel_sizes(1.0, 1.0 + 1e-8)
    assert validate_voxel_sizes(1e-6, 0.0)  # within rtol
    assert not validate_voxel_sizes(1.0, 1.0 + 1e-3)

