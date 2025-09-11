#!/usr/bin/env python3
import types
import numpy as np


def test_marker_array_to_voxel_grid_uses_pcd_only(monkeypatch=None):
    # Build a minimal open3d stub
    o3d_stub = types.SimpleNamespace()
    class _VoxelGrid:
        pass
    class _GeomNS:
        class VoxelGrid:
            @staticmethod
            def create_from_point_cloud(pcd, voxel_size=0.1):
                # Assert pcd is the object we passed (not a tuple)
                assert hasattr(pcd, 'is_pcd') and pcd.is_pcd is True
                return _VoxelGrid()
        class PointCloud:
            def __init__(self):
                self.is_pcd = True
    o3d_stub.geometry = _GeomNS
    o3d_stub.utility = types.SimpleNamespace(Vector3dVector=lambda a: a)

    import sys
    sys.modules['open3d'] = o3d_stub

    from voxel_viewer.marker_to_open3d import MarkerToOpen3D

    # Monkeypatch marker_array_to_pointcloud to return (pcd, scale)
    conv = MarkerToOpen3D()
    def fake_mapc(marker_array):
        return o3d_stub.geometry.PointCloud(), 0.5
    conv.marker_array_to_pointcloud = fake_mapc

    class _DummyMA:
        markers = []

    vg = conv.marker_array_to_voxel_grid(_DummyMA(), voxel_size=0.2)
    assert isinstance(vg, _VoxelGrid)

